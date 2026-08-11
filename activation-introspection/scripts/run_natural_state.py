"""Pilot: can opaque-code ICL report a naturally computed route state?"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

# Load the 3B checkpoint directly at its inference dtype. The default MPS
# watermark otherwise permits swap-heavy runs instead of failing early.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import models
from introspect.codebook_icl import LABELS, exact_episodes, prepare_episode, sha256_text
from introspect.hooks import capture
from introspect.natural_state import (
    ROUTE_WORLDS,
    RouteWorld,
    patch_residuals,
    route_bridge,
    unique_substring_token_position,
)
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
ANSWER_PREFIX = "Final:"
VISIBLE_SAMPLE = "A neutral route worksheet is open. Hidden state marker: §"
CONDITIONS = ("clean", "sham", "query_only", "natural", "anti_grounded")
SOURCE_PATHS = (
    "scripts/run_natural_state.py",
    "src/introspect/natural_state.py",
    "src/introspect/codebook_icl.py",
    "src/introspect/hooks.py",
    "src/introspect/models.py",
    "src/introspect/preflight.py",
    "pyproject.toml",
    "uv.lock",
)


@dataclass(frozen=True)
class Donor:
    world: RouteWorld
    sign: int
    prompt: str
    input_ids: Tensor
    marker_position: int
    state: Tensor
    logits: Tensor
    answer_ids: tuple[int, int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _source_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load() -> models.LoadedModel:
    """Load once, directly onto MPS, without the transient CPU copy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = models.KNOWN_MODELS[MODEL]
    tokenizer = cast(Any, AutoTokenizer).from_pretrained(repo, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    module = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="mps",
        low_cpu_mem_usage=True,
    )
    cast(Any, module).eval()
    module.requires_grad_(False)
    return models.LoadedModel(
        name=repo,
        model=module,
        tokenizer=tokenizer,
        device=torch.device("mps"),
        dtype=torch.bfloat16,
    )


def _single_continuation_id(model: models.LoadedModel, prompt: str, text: str) -> int:
    before = model.encode(prompt)
    after = model.encode(prompt + " " + text)
    if int(after.shape[1]) != int(before.shape[1]) + 1 or not torch.equal(after[:, :-1], before):
        raise ValueError(f"{text!r} is not one continuation token")
    return int(after[0, -1])


def _score(
    logits: Tensor, labels: tuple[str, str], token_ids: tuple[int, int]
) -> dict[str, object]:
    value = logits.float()
    candidates = torch.tensor(token_ids, device=value.device)
    selected = value[candidates]
    conditional = torch.softmax(selected, dim=-1)
    full = torch.log_softmax(value, dim=-1)
    predicted = labels[int(selected.argmax())]
    return {
        "predicted": predicted,
        "conditional_probs": {label: float(conditional[i]) for i, label in enumerate(labels)},
        "full_logprobs": {
            label: float(full[token_id]) for label, token_id in zip(labels, token_ids, strict=True)
        },
        "label_mass": float(torch.logsumexp(selected, 0).sub(torch.logsumexp(value, 0)).exp()),
        "format_ok": int(value.argmax()) in set(token_ids),
        "top_token_id": int(value.argmax()),
    }


@torch.no_grad()
def _prepare_donor(model: models.LoadedModel, world: RouteWorld, sign: int) -> Donor:
    bridge = route_bridge(sign)
    prompt = model.chat(world.render_user(bridge), assistant_prefix=ANSWER_PREFIX)
    input_ids = model.encode(prompt)
    marker_position = unique_substring_token_position(model.tokenizer, prompt, "§")
    answer_ids = (
        _single_continuation_id(model, prompt, world.endpoint(route_bridge(1))),
        _single_continuation_id(model, prompt, world.endpoint(route_bridge(-1))),
    )
    with capture(model, [LAYER]) as store:
        logits = model.forward_logits(input_ids)[0, -1].float().cpu()
    state = store.acts[LAYER][0][0, marker_position].clone()
    return Donor(
        world=world,
        sign=sign,
        prompt=prompt,
        input_ids=input_ids,
        marker_position=marker_position,
        state=state,
        logits=logits,
        answer_ids=answer_ids,
    )


@torch.no_grad()
def _patched_logits(
    model: models.LoadedModel,
    input_ids: Tensor,
    positions: tuple[int, ...],
    states: tuple[Tensor, ...],
    expected_recipients: tuple[Tensor, ...],
) -> Tensor:
    stacked = torch.stack(states)
    with (
        patch_residuals(
            model,
            LAYER,
            positions,
            stacked,
            expected_recipients=torch.stack(expected_recipients),
        ),
        capture(model, [LAYER]) as seen,
    ):
        logits = model.forward_logits(input_ids)[0, -1].float().cpu()
    actual = seen.acts[LAYER][0][0]
    for position, expected in zip(positions, states, strict=True):
        if not torch.allclose(actual[position], expected.float(), atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"replacement drift at token {position}")
    return logits


def _answer_score(donor: Donor, logits: Tensor, expected_sign: int) -> dict[str, object]:
    labels = (
        donor.world.endpoint(route_bridge(1)),
        donor.world.endpoint(route_bridge(-1)),
    )
    score = _score(logits, labels, donor.answer_ids)
    expected = donor.world.endpoint(route_bridge(expected_sign))
    score["expected"] = expected
    score["correct"] = score["predicted"] == expected
    return score


def _signed_margin(donor: Donor, logits: Tensor, sign: int) -> float:
    target = donor.answer_ids[0 if sign == 1 else 1]
    other = donor.answer_ids[1 if sign == 1 else 0]
    return float(logits[target] - logits[other])


def _normalized_recovery(recipient_clean: float, patched: float, donor_clean: float) -> float:
    denominator = donor_clean - recipient_clean
    if abs(denominator) <= 1e-6:
        raise ValueError("normalized recovery denominator is too small")
    return (patched - recipient_clean) / denominator


def _protocol(smoke: bool) -> dict[str, object]:
    review_prompt = (
        "Act as a hostile experimental-design reviewer. We have Qwen2.5-3B, "
        "residual-stream capture/replacement hooks, and an existing four-shot "
        "episode-remapped Q/K benchmark with byte-identical visible observations. "
        "We want the cheapest inference-only pilot testing whether the model can "
        "report which of two naturally computed hidden states was activation-patched "
        "into identical report prompts. Propose exactly one minimal donor task and "
        "patch/readout schedule. Require: (1) a causal reachability positive control "
        "showing the donor state affects ordinary behavior, (2) a sham or irrelevant-"
        "donor control, (3) unrestricted next-token format scoring, (4) disjoint DEV "
        "selection and a frozen test, and (5) an explicit kill rule. Identify the "
        "strongest alternative explanation that would remain after a positive. Keep "
        "the answer under 350 words and do not use tools."
    )
    return {
        "schema_version": 1,
        "frozen_on": "2026-08-10",
        "disclosed_precursor": (
            "Smoke protocol v1 stopped before writing raw data: Intervention(mode='replace') "
            "renormalized the donor tensor at bfloat16 and failed the exact-replacement "
            "assertion. V2 changes only the replacement implementation and recipient guard."
        ),
        "question": (
            "Can episode-remapped ICL classify which naturally generated, causally "
            "load-bearing two-hop route state was transplanted into a matched prompt?"
        ),
        "claim_boundary": (
            "A positive is causal reportability of a transplanted route state under "
            "one model/layer/interface. It is not privileged introspection, a pure "
            "bridge circuit, natural free-form verbalization, or population evidence."
        ),
        "competing_explanations": [
            "the report reads a low-level donor signature rather than route content",
            "the ordinary answer is not causally reachable from the patched marker",
            "the model follows visible label regularities rather than hidden states",
        ],
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "device": "mps",
            "layer": LAYER,
            "bridges": {"+1": "maple", "-1": "cedar"},
            "worlds": [asdict(world) for world in ROUTE_WORLDS],
            "visible_sample": VISIBLE_SAMPLE,
            "conditions": list(CONDITIONS),
            "cells_per_fold": 4 if smoke else 24,
            "folds": "five leave-one-world-out folds; four donor worlds and one query world",
            "labels": list(LABELS),
            "smoke": smoke,
        },
        "reachability_gate": {
            "clean_full_vocab_answers_correct": "10/10",
            "self_patch_max_abs_logit_error": 1e-4,
            "bidirectional_cross_patch_worlds": "at least 4/5",
            "mean_normalized_log_odds_recovery": 0.5,
            "recovery_estimand": (
                "(patched_margin - recipient_clean_margin) / "
                "(donor_clean_margin - recipient_clean_margin); "
                "fail if abs(denominator) <= 1e-6"
            ),
            "stop": "do not run or rescue the reporter if this gate fails",
        },
        "report_gates": {
            "natural_accuracy": 0.75,
            "natural_minus_query_only": 0.20,
            "worlds_with_positive_difference": "at least 4/5",
            "natural_query_twin_both_correct": 0.60,
            "anti_grounded_inverse_accuracy": 0.75,
            "natural_format_rate": 0.90,
            "natural_mean_label_mass": 0.50,
            "sham_matches_clean": "max Q/K log-probability difference <= 1e-4",
            "stop": "report the frozen null; do not change layer, worlds, prompt, or gates",
        },
        "analysis": (
            "world is the unit; the exact order x mapping x query cells are nuisance "
            "marginalization. Report every world and paired statistic."
        ),
        "agent_harness": {
            "runner": "codex exec",
            "cli": "codex-cli 0.146.0",
            "model": "gpt-5.6-luna",
            "flags": "--ignore-user-config --ignore-rules --ephemeral --sandbox read-only",
            "requested_reasoning": "minimal (rejected as unsupported)",
            "effective_reasoning": "low",
            "fallback": None,
            "system_prompt": "stock Codex system prompt; not exposed by the CLI",
            "user_prompt_prefix": review_prompt,
        },
        "source_files_sha256": _source_hashes(),
    }


def _freeze_protocol(path: Path, smoke: bool) -> tuple[dict[str, object], str]:
    protocol = _protocol(smoke)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit(f"{path} differs from this source; issue a new protocol version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


def _write_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _reachability(
    model: models.LoadedModel,
    donors: dict[tuple[str, int], Donor],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    clean_ok = True
    self_errors: list[float] = []
    cross_recoveries: list[float] = []
    cross_worlds = 0
    for world in ROUTE_WORLDS:
        world_cross_ok = True
        for recipient_sign in (1, -1):
            recipient = donors[(world.start, recipient_sign)]
            clean_score = _answer_score(recipient, recipient.logits, recipient_sign)
            clean_ok &= bool(clean_score["correct"] and clean_score["format_ok"])
            rows.append(
                {
                    "record_type": "donor_clean",
                    "world": world.start,
                    "sign": recipient_sign,
                    "prompt": recipient.prompt,
                    "prompt_sha256": sha256_text(recipient.prompt),
                    "token_ids": recipient.input_ids[0].tolist(),
                    "marker_position": recipient.marker_position,
                    "state_sha256": _json_sha256(recipient.state.tolist()),
                    "state_norm": float(recipient.state.norm()),
                    "score": clean_score,
                }
            )
            for donor_sign in (recipient_sign, -recipient_sign):
                state = donors[(world.start, donor_sign)].state
                logits = _patched_logits(
                    model,
                    recipient.input_ids,
                    (recipient.marker_position,),
                    (state,),
                    (recipient.state,),
                )
                score = _answer_score(recipient, logits, donor_sign)
                self_patch = donor_sign == recipient_sign
                max_error = float((logits - recipient.logits).abs().max()) if self_patch else None
                recovery = None
                if self_patch:
                    self_errors.append(cast(float, max_error))
                else:
                    donor_clean = donors[(world.start, donor_sign)]
                    recovery = _normalized_recovery(
                        _signed_margin(recipient, recipient.logits, donor_sign),
                        _signed_margin(recipient, logits, donor_sign),
                        _signed_margin(donor_clean, donor_clean.logits, donor_sign),
                    )
                    cross_recoveries.append(recovery)
                    world_cross_ok &= bool(score["correct"] and score["format_ok"])
                rows.append(
                    {
                        "record_type": "reach_patch",
                        "world": world.start,
                        "recipient_sign": recipient_sign,
                        "donor_sign": donor_sign,
                        "self_patch": self_patch,
                        "max_abs_logit_error": max_error,
                        "normalized_log_odds_recovery": recovery,
                        "score": score,
                    }
                )
        cross_worlds += int(world_cross_ok)
    mean_recovery = sum(cross_recoveries) / len(cross_recoveries)
    max_self_error = max(self_errors)
    passed = clean_ok and max_self_error <= 1e-4 and cross_worlds >= 4 and mean_recovery >= 0.5
    return {
        "clean_full_vocab_answers_correct": clean_ok,
        "self_patch_max_abs_logit_error": max_self_error,
        "bidirectional_cross_patch_worlds": cross_worlds,
        "mean_normalized_log_odds_recovery": mean_recovery,
        "passed": passed,
    }


def _label_score(
    logits: Tensor, label_ids: tuple[int, int], correct_label: str
) -> dict[str, object]:
    score = _score(logits, LABELS, label_ids)
    score["correct_target"] = score["predicted"] == correct_label
    score["correct_inverse"] = score["predicted"] != correct_label
    return score


@torch.no_grad()
def _report(
    model: models.LoadedModel,
    donors: dict[tuple[str, int], Donor],
    *,
    smoke: bool,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    report_rows: list[dict[str, object]] = []
    episodes = exact_episodes(VISIBLE_SAMPLE)
    if smoke:
        episodes = episodes[:4]
    for fold, query_world in enumerate(ROUTE_WORLDS):
        demo_worlds = tuple(world for world in ROUTE_WORLDS if world != query_world)
        for episode in episodes:
            prepared = prepare_episode(model, episode)
            with capture(model, [LAYER]) as clean_capture:
                clean_logits = model.forward_logits(prepared.input_ids)[0, -1].float().cpu()
            clean_states = tuple(
                clean_capture.acts[LAYER][0][0, position].clone()
                for position in prepared.state_positions
            )
            target_states = (
                *(
                    donors[(world.start, sign)].state
                    for world, sign in zip(demo_worlds, episode.demo_signs, strict=True)
                ),
                donors[(query_world.start, episode.query_sign)].state,
            )
            anti_states = (
                *(
                    donors[(world.start, -sign)].state
                    for world, sign in zip(demo_worlds, episode.demo_signs, strict=True)
                ),
                donors[(query_world.start, episode.query_sign)].state,
            )
            logits_by_condition = {
                "clean": clean_logits,
                "sham": _patched_logits(
                    model,
                    prepared.input_ids,
                    prepared.state_positions,
                    clean_states,
                    clean_states,
                ),
                "query_only": _patched_logits(
                    model,
                    prepared.input_ids,
                    (prepared.state_positions[-1],),
                    (target_states[-1],),
                    (clean_states[-1],),
                ),
                "natural": _patched_logits(
                    model,
                    prepared.input_ids,
                    prepared.state_positions,
                    target_states,
                    clean_states,
                ),
                "anti_grounded": _patched_logits(
                    model,
                    prepared.input_ids,
                    prepared.state_positions,
                    anti_states,
                    clean_states,
                ),
            }
            condition_scores = {
                condition: _label_score(logits, prepared.label_ids, episode.correct_label)
                for condition, logits in logits_by_condition.items()
            }
            row: dict[str, object] = {
                "record_type": "report",
                "fold": fold,
                "query_world": query_world.start,
                "demo_worlds": [world.start for world in demo_worlds],
                "cell_id": episode.cell_id,
                "demo_signs": list(episode.demo_signs),
                "query_sign": episode.query_sign,
                "positive_label": episode.positive_label,
                "negative_label": episode.negative_label,
                "correct_label": episode.correct_label,
                "prompt": prepared.prompt,
                "prompt_sha256": prepared.prompt_sha256,
                "token_ids": prepared.input_ids[0].tolist(),
                "state_positions": list(prepared.state_positions),
                "condition_scores": condition_scores,
            }
            rows.append(row)
            report_rows.append(row)
        print(f"report fold {fold + 1}/5 {query_world.start}", flush=True)
    return report_rows


def _condition_accuracy(rows: list[dict[str, object]], condition: str, key: str) -> float:
    return sum(
        bool(cast(dict[str, Any], row["condition_scores"])[condition][key]) for row in rows
    ) / len(rows)


def _paired_accuracy(rows: list[dict[str, object]], *, mapping_flip: bool = False) -> float:
    groups: dict[tuple[object, ...], list[bool]] = {}
    for row in rows:
        if mapping_flip:
            key = (row["query_world"], tuple(cast(list[int], row["demo_signs"])), row["query_sign"])
        else:
            key = (
                row["query_world"],
                tuple(cast(list[int], row["demo_signs"])),
                row["positive_label"],
            )
        natural = cast(dict[str, Any], row["condition_scores"])["natural"]
        groups.setdefault(key, []).append(bool(natural["correct_target"]))
    complete = [values for values in groups.values() if len(values) == 2]
    return sum(all(values) for values in complete) / len(complete)


def _report_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    natural = _condition_accuracy(rows, "natural", "correct_target")
    query_only = _condition_accuracy(rows, "query_only", "correct_target")
    anti_inverse = _condition_accuracy(rows, "anti_grounded", "correct_inverse")
    by_world: dict[str, dict[str, float]] = {}
    for world in ROUTE_WORLDS:
        subset = [row for row in rows if row["query_world"] == world.start]
        by_world[world.start] = {
            "natural_accuracy": _condition_accuracy(subset, "natural", "correct_target"),
            "query_only_accuracy": _condition_accuracy(subset, "query_only", "correct_target"),
        }
    positive_worlds = sum(
        values["natural_accuracy"] > values["query_only_accuracy"] for values in by_world.values()
    )
    natural_scores = [cast(dict[str, Any], row["condition_scores"])["natural"] for row in rows]
    sham_errors = []
    for row in rows:
        scores = cast(dict[str, Any], row["condition_scores"])
        for label in LABELS:
            sham_errors.append(
                abs(
                    scores["sham"]["full_logprobs"][label] - scores["clean"]["full_logprobs"][label]
                )
            )
    metrics = {
        "natural_accuracy": natural,
        "query_only_accuracy": query_only,
        "natural_minus_query_only": natural - query_only,
        "worlds_with_positive_difference": positive_worlds,
        "by_world": by_world,
        "natural_query_twin_both_correct": _paired_accuracy(rows),
        "natural_mapping_flip_both_correct": _paired_accuracy(rows, mapping_flip=True),
        "anti_grounded_inverse_accuracy": anti_inverse,
        "natural_format_rate": sum(bool(score["format_ok"]) for score in natural_scores)
        / len(natural_scores),
        "natural_mean_label_mass": sum(float(score["label_mass"]) for score in natural_scores)
        / len(natural_scores),
        "sham_clean_max_label_logprob_difference": max(sham_errors),
    }
    gates = {
        "natural_accuracy": natural >= 0.75,
        "natural_minus_query_only": natural - query_only >= 0.20,
        "worlds_with_positive_difference": positive_worlds >= 4,
        "natural_query_twin_both_correct": metrics["natural_query_twin_both_correct"] >= 0.60,
        "anti_grounded_inverse_accuracy": anti_inverse >= 0.75,
        "natural_format_rate": metrics["natural_format_rate"] >= 0.90,
        "natural_mean_label_mass": metrics["natural_mean_label_mass"] >= 0.50,
        "sham_matches_clean": max(sham_errors) <= 1e-4,
    }
    return {"metrics": metrics, "gates": gates, "all_gates_pass": all(gates.values())}


def run(args: argparse.Namespace) -> None:
    out = args.out or _default_output(args.smoke)
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    protocol_path = args.protocol or Path(
        "results/natural_state_smoke_protocol_v2.json"
        if args.smoke
        else "results/natural_state_dev_protocol_v1.json"
    )
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _freeze_protocol(protocol_path, args.smoke)
    preflight_check(MODEL, training=False)
    model = _load()
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        if models.loaded_revision(model) != MODEL_REVISION:
            raise SystemExit("loaded model revision does not match the frozen revision")
        donors = {
            (world.start, sign): _prepare_donor(model, world, sign)
            for world in ROUTE_WORLDS
            for sign in (1, -1)
        }
        reachability = _reachability(model, donors, rows)
        report_rows = (
            _report(model, donors, smoke=args.smoke, rows=rows) if reachability["passed"] else []
        )
        report = _report_summary(report_rows) if report_rows else None
        summary = {
            "schema_version": 1,
            "reachability": reachability,
            "report": report,
            "status": (
                "pilot_pass"
                if report is not None and report["all_gates_pass"]
                else "stop_reporter_gate_failed"
                if report is not None
                else "stop_reachability_failed"
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out, rows)
        raw_sha = _sha256(out)
        config = {
            "schema_version": 1,
            "model": model.name,
            "model_revision": models.loaded_revision(model),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "layer": LAYER,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        manifest = {
            "schema_version": 1,
            "config": config,
            "config_sha256": _json_sha256(config),
            "raw": out.name,
            "raw_sha256": raw_sha,
            "n_rows": len(rows),
            "n_report_rows": len(report_rows),
            "elapsed_seconds": time.time() - started,
        }
        _write_json(manifest_path, manifest)
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {out} ({raw_sha})", flush=True)
    finally:
        model.free()


def _default_output(smoke: bool) -> Path:
    return Path(
        "results/natural_state_smoke_v3_raw.jsonl"
        if smoke
        else "results/natural_state_dev_v1_raw.jsonl"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
