"""Pilot: can opaque-code ICL report a naturally computed, output-ready state?

The two-hop route pilot (``notes/09``) stopped at its reachability gate: the
layer-9 route marker did not control the ordinary answer, so the reporter never
ran. This runner keeps that gate and changes the state being transplanted. The
donor is now the residual at the last pre-answer token of a single-digit
arithmetic problem the model solves itself, screened over three prospectively
named anchor layers. The hidden class is the parity of the answer.
"""

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
from introspect.codebook_icl import ANSWER_PREFIX as LABEL_PREFIX
from introspect.codebook_icl import (
    LABELS,
    Episode,
    exact_episodes,
    prepare_episode,
    sha256_text,
)
from introspect.hooks import capture
from introspect.natural_state import (
    ARITH_DEV,
    ARITH_TEST,
    ArithTask,
    patch_residuals,
)
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
#: Prospectively named anchors. 9 is the layer every prior reporting result on
#: this model used; 21 and 26 are the query-marker and final-answer sites the
#: Stage 1b localization screen selected. The earliest that passes is used.
ANCHOR_LAYERS = (9, 21, 26)
#: The trailing space is deliberate. Qwen2 merges " Q" into one token but splits
#: " 8" into two, so the prefix has to carry the space for a one-token answer.
ANSWER_PREFIX = "Answer: "
VISIBLE_SAMPLE = "A neutral status note is displayed. Hidden state marker: §"
CONDITIONS = ("clean", "sham", "query_only", "natural", "anti_grounded", "visible")
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
    task: ArithTask
    sign: int
    prompt: str
    input_ids: Tensor
    position: int
    states: dict[int, Tensor]
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
    """Token id of ``text`` as one continuation; the caller owns any leading space."""
    before = model.encode(prompt)
    after = model.encode(prompt + text)
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


def _anchors(site: int | None) -> tuple[int, ...]:
    """The layers this run screens: the blind anchor set, or one named site."""
    return ANCHOR_LAYERS if site is None else (site,)


@torch.no_grad()
def _prepare_donor(
    model: models.LoadedModel, task: ArithTask, sign: int, anchors: tuple[int, ...]
) -> Donor:
    """Solve one problem cleanly and keep the last pre-answer residual."""
    prompt = model.chat(task.render_user(sign), assistant_prefix=ANSWER_PREFIX)
    input_ids = model.encode(prompt)
    position = int(input_ids.shape[1]) - 1
    answer_ids = (
        _single_continuation_id(model, prompt, str(task.answer(1))),
        _single_continuation_id(model, prompt, str(task.answer(-1))),
    )
    with capture(model, list(anchors)) as store:
        logits = model.forward_logits(input_ids)[0, -1].float().cpu()
    states = {layer: store.acts[layer][0][0, position].clone() for layer in anchors}
    return Donor(
        task=task,
        sign=sign,
        prompt=prompt,
        input_ids=input_ids,
        position=position,
        states=states,
        logits=logits,
        answer_ids=answer_ids,
    )


@torch.no_grad()
def _patched_logits(
    model: models.LoadedModel,
    layer: int,
    input_ids: Tensor,
    positions: tuple[int, ...],
    states: tuple[Tensor, ...],
    expected_recipients: tuple[Tensor, ...],
) -> Tensor:
    stacked = torch.stack(states)
    with (
        patch_residuals(
            model,
            layer,
            positions,
            stacked,
            expected_recipients=torch.stack(expected_recipients),
        ),
        capture(model, [layer]) as seen,
    ):
        logits = model.forward_logits(input_ids)[0, -1].float().cpu()
    actual = seen.acts[layer][0][0]
    for position, expected in zip(positions, states, strict=True):
        if not torch.allclose(actual[position], expected.float(), atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"replacement drift at token {position}")
    return logits


def _answer_labels(donor: Donor) -> tuple[str, str]:
    return (str(donor.task.answer(1)), str(donor.task.answer(-1)))


def _answer_score(donor: Donor, logits: Tensor, expected_sign: int) -> dict[str, object]:
    labels = _answer_labels(donor)
    score = _score(logits, labels, donor.answer_ids)
    expected = str(donor.task.answer(expected_sign))
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


def _protocol(smoke: bool, site: int | None) -> dict[str, object]:
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
        "schema_version": 2,
        "frozen_on": "2026-08-11",
        "disclosed_precursor": (
            "natural_state_smoke_protocol_v2 transplanted a two-hop route marker at layer 9 "
            "and stopped: clean answers 8/10 and bidirectional cross-patching 0/5. This "
            "protocol changes the donor state, the capture site, and the layer screen. It "
            "reuses that runner's exact-replacement hook, gate algebra, and stop rule."
        )
        if site is None
        else (
            "natural_state_arith_smoke_protocol_v1 screened blind anchors 9, 21 and 26 with "
            "these exact stimuli and gates, and stopped: cross-patching controlled 0/5 tasks "
            "at every anchor. A post-hoc all-layer localization on the development bank "
            "(natural_state_arith_site_diagnostic_v1.json) then showed the pre-answer state "
            "does not carry the answer below block 27 and controls it in 10/10 transplants "
            "from block 27. THE SITE BELOW IS THAT POST-HOC SELECTION, NOT A BLIND ONE. "
            "Nothing else changes: the same stimuli, banks, conditions, gates and stop rule. "
            "The held-out bank has never been scored, in that run or the diagnostic."
        ),
        "question": (
            "Can episode-remapped ICL classify the parity of a naturally computed, "
            "causally load-bearing arithmetic answer state transplanted into a matched "
            "report prompt?"
        ),
        "claim_boundary": (
            "A positive is causal reportability of a transplanted output-ready state "
            "under one model, one selected layer, and one interface. It is deliberately "
            "narrower than reporting a hidden intermediate: the donor state is the one "
            "the model was about to act on. It is not privileged introspection, natural "
            "free-form verbalization, or population evidence."
        )
        + (
            ""
            if site is None
            else (
                " The site is a post-hoc development selection, so a positive is a "
                "single-site demonstration rather than a blind confirmation, and the "
                "state at this depth already favours its own answer under a logit lens."
            )
        ),
        "competing_explanations": [
            "the report reads the donor answer token's identity rather than its parity",
            "the ordinary answer is not causally reachable from the patched position",
            "the model follows visible label regularities rather than hidden states",
            "a null reflects failure to induce the parity rule, not failure to read the "
            "state; the visible control exists to separate these and gates the reading",
        ]
        + (
            []
            if site is None
            else [
                "the transplanted state is close to the answer token itself, so a positive "
                "shows reporting of an output-ready state and not of a hidden intermediate"
            ]
        ),
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "device": "mps",
            "anchor_layers": list(_anchors(site)),
            "layer_selection": (
                "screen every anchor on the DEV bank; use the earliest that passes the "
                "reachability gate; stop if none does"
            )
            if site is None
            else (
                f"block {site}, selected post-hoc from the development-bank localization "
                "in natural_state_arith_site_diagnostic_v1.json. It is re-confirmed on the "
                "development bank here and must independently pass the same gate on the "
                "held-out bank before any reporting row runs. No other layer is tested, "
                "and no reselection is permitted if it fails."
            ),
            "capture_site": "last pre-answer token of the clean arithmetic prompt",
            "hidden_class": {"+1": "even answer", "-1": "odd answer"},
            "answer_prefix": ANSWER_PREFIX,
            "dev_tasks": [asdict(task) for task in ARITH_DEV],
            "test_tasks": [asdict(task) for task in ARITH_TEST],
            "visible_sample": VISIBLE_SAMPLE,
            "conditions": list(CONDITIONS),
            "cells_per_fold": 4 if smoke else 24,
            "folds": "five leave-one-task-out folds over the held-out bank",
            "labels": list(LABELS),
            "smoke": smoke,
        },
        "reachability_gate": {
            "applies_to": "each anchor layer on DEV, then the selected layer on the test bank",
            "clean_full_vocab_answers_correct": "10/10",
            "self_patch_max_abs_logit_error": 1e-4,
            "bidirectional_cross_patch_tasks": "at least 4/5",
            "mean_normalized_log_odds_recovery": 0.5,
            "recovery_estimand": (
                "(patched_margin - recipient_clean_margin) / "
                "(donor_clean_margin - recipient_clean_margin); "
                "fail if abs(denominator) <= 1e-6"
            ),
            "stop": (
                "no anchor passes on DEV: close the single-position natural-state "
                "transplant family. Selected layer fails on the test bank: stop without "
                "reporting rather than reselecting."
            ),
        },
        "report_gates": {
            "natural_accuracy": 0.75,
            "natural_minus_query_only": 0.20,
            "tasks_with_positive_difference": "at least 4/5",
            "natural_query_twin_both_correct": 0.60,
            "anti_grounded_inverse_accuracy": 0.75,
            "natural_format_rate": 0.90,
            "natural_mean_label_mass": 0.50,
            "sham_matches_clean": "max Q/K log-probability difference <= 1e-4",
            "stop": "report the frozen null; do not change layer, tasks, prompt, or gates",
        },
        "interpretation_gate": {
            "visible_accuracy": 0.75,
            "role": (
                "capability control, not a reporting gate. The same episodes with the "
                "arithmetic problems written out and no patching. A natural-state null "
                "is a reporting null only if this passes; otherwise it is a failure to "
                "induce the parity rule and the pilot reports an instrument failure."
            ),
        },
        "analysis": (
            "task is the unit; the exact order x mapping x query cells are nuisance "
            "marginalization. Report every task and paired statistic."
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


def _freeze_protocol(path: Path, smoke: bool, site: int | None) -> tuple[dict[str, object], str]:
    protocol = _protocol(smoke, site)
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


def _clean_gate(
    donors: dict[tuple[str, int], Donor],
    tasks: tuple[ArithTask, ...],
    bank: str,
    rows: list[dict[str, object]],
) -> bool:
    """Score the unpatched answers once; they do not depend on the anchor layer."""
    ok = True
    for task in tasks:
        for sign in (1, -1):
            donor = donors[(task.name, sign)]
            score = _answer_score(donor, donor.logits, sign)
            ok &= bool(score["correct"] and score["format_ok"])
            rows.append(
                {
                    "record_type": "donor_clean",
                    "bank": bank,
                    "task": task.name,
                    "problem": task.problem(sign),
                    "sign": sign,
                    "prompt": donor.prompt,
                    "prompt_sha256": sha256_text(donor.prompt),
                    "token_ids": donor.input_ids[0].tolist(),
                    "position": donor.position,
                    "state_sha256": {
                        str(layer): _json_sha256(state.tolist())
                        for layer, state in donor.states.items()
                    },
                    "state_norm": {
                        str(layer): float(state.norm()) for layer, state in donor.states.items()
                    },
                    "score": score,
                }
            )
    return ok


def _reachability(
    model: models.LoadedModel,
    donors: dict[tuple[str, int], Donor],
    tasks: tuple[ArithTask, ...],
    layer: int,
    bank: str,
    clean_ok: bool,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Does replacing this position's state make the ordinary answer follow the donor?"""
    self_errors: list[float] = []
    cross_recoveries: list[float] = []
    cross_tasks = 0
    for task in tasks:
        task_cross_ok = True
        for recipient_sign in (1, -1):
            recipient = donors[(task.name, recipient_sign)]
            for donor_sign in (recipient_sign, -recipient_sign):
                state = donors[(task.name, donor_sign)].states[layer]
                logits = _patched_logits(
                    model,
                    layer,
                    recipient.input_ids,
                    (recipient.position,),
                    (state,),
                    (recipient.states[layer],),
                )
                score = _answer_score(recipient, logits, donor_sign)
                self_patch = donor_sign == recipient_sign
                max_error = float((logits - recipient.logits).abs().max()) if self_patch else None
                recovery = None
                if self_patch:
                    self_errors.append(cast(float, max_error))
                else:
                    donor_clean = donors[(task.name, donor_sign)]
                    recovery = _normalized_recovery(
                        _signed_margin(recipient, recipient.logits, donor_sign),
                        _signed_margin(recipient, logits, donor_sign),
                        _signed_margin(donor_clean, donor_clean.logits, donor_sign),
                    )
                    cross_recoveries.append(recovery)
                    task_cross_ok &= bool(score["correct"] and score["format_ok"])
                rows.append(
                    {
                        "record_type": "reach_patch",
                        "bank": bank,
                        "layer": layer,
                        "task": task.name,
                        "recipient_sign": recipient_sign,
                        "donor_sign": donor_sign,
                        "self_patch": self_patch,
                        "max_abs_logit_error": max_error,
                        "normalized_log_odds_recovery": recovery,
                        "score": score,
                    }
                )
        cross_tasks += int(task_cross_ok)
    mean_recovery = sum(cross_recoveries) / len(cross_recoveries)
    max_self_error = max(self_errors)
    return {
        "layer": layer,
        "bank": bank,
        "clean_full_vocab_answers_correct": clean_ok,
        "self_patch_max_abs_logit_error": max_self_error,
        "bidirectional_cross_patch_tasks": cross_tasks,
        "mean_normalized_log_odds_recovery": mean_recovery,
        "passed": clean_ok and max_self_error <= 1e-4 and cross_tasks >= 4 and mean_recovery >= 0.5,
    }


def _select_layer(
    screen: dict[int, dict[str, object]], anchors: tuple[int, ...] = ANCHOR_LAYERS
) -> int | None:
    """Earliest anchor passing the frozen reachability gate, as pre-registered."""
    for layer in anchors:
        if screen[layer]["passed"]:
            return layer
    return None


def _label_score(
    logits: Tensor, label_ids: tuple[int, int], correct_label: str
) -> dict[str, object]:
    score = _score(logits, LABELS, label_ids)
    score["correct_target"] = score["predicted"] == correct_label
    score["correct_inverse"] = score["predicted"] != correct_label
    return score


def _visible_user(
    episode: Episode, demo_tasks: tuple[ArithTask, ...], query_task: ArithTask
) -> str:
    """The same induction problem with the arithmetic written out and nothing patched."""
    lines = [
        "Infer the mapping from a hidden property of each result to the opaque labels Q and K.",
        "Every observation is one arithmetic problem.",
        "Use the demonstrations, then answer the held-out query with one label.",
    ]
    for task, sign in zip(demo_tasks, episode.demo_signs, strict=True):
        lines.extend(
            [
                "",
                "Demonstration:",
                f"Observation: {task.problem(sign)}",
                f"Label: {episode.label_for(sign)}",
            ]
        )
    lines.extend(["", "Held-out query:", f"Observation: {query_task.problem(episode.query_sign)}"])
    return "\n".join(lines)


@torch.no_grad()
def _visible_score(
    model: models.LoadedModel,
    episode: Episode,
    demo_tasks: tuple[ArithTask, ...],
    query_task: ArithTask,
) -> tuple[str, dict[str, object]]:
    prompt = model.chat(_visible_user(episode, demo_tasks, query_task), LABEL_PREFIX)
    label_ids = cast(
        tuple[int, int],
        tuple(_single_continuation_id(model, prompt, f" {label}") for label in LABELS),
    )
    logits = model.forward_logits(model.encode(prompt))[0, -1].float().cpu()
    return prompt, _label_score(logits, label_ids, episode.correct_label)


@torch.no_grad()
def _report(
    model: models.LoadedModel,
    donors: dict[tuple[str, int], Donor],
    tasks: tuple[ArithTask, ...],
    layer: int,
    *,
    smoke: bool,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    report_rows: list[dict[str, object]] = []
    episodes = exact_episodes(VISIBLE_SAMPLE)
    if smoke:
        episodes = episodes[:4]
    for fold, query_task in enumerate(tasks):
        demo_tasks = tuple(task for task in tasks if task != query_task)
        for episode in episodes:
            prepared = prepare_episode(model, episode)
            with capture(model, [layer]) as clean_capture:
                clean_logits = model.forward_logits(prepared.input_ids)[0, -1].float().cpu()
            clean_states = tuple(
                clean_capture.acts[layer][0][0, position].clone()
                for position in prepared.state_positions
            )
            target_states = (
                *(
                    donors[(task.name, sign)].states[layer]
                    for task, sign in zip(demo_tasks, episode.demo_signs, strict=True)
                ),
                donors[(query_task.name, episode.query_sign)].states[layer],
            )
            anti_states = (
                *(
                    donors[(task.name, -sign)].states[layer]
                    for task, sign in zip(demo_tasks, episode.demo_signs, strict=True)
                ),
                donors[(query_task.name, episode.query_sign)].states[layer],
            )
            logits_by_condition = {
                "clean": clean_logits,
                "sham": _patched_logits(
                    model,
                    layer,
                    prepared.input_ids,
                    prepared.state_positions,
                    clean_states,
                    clean_states,
                ),
                "query_only": _patched_logits(
                    model,
                    layer,
                    prepared.input_ids,
                    (prepared.state_positions[-1],),
                    (target_states[-1],),
                    (clean_states[-1],),
                ),
                "natural": _patched_logits(
                    model,
                    layer,
                    prepared.input_ids,
                    prepared.state_positions,
                    target_states,
                    clean_states,
                ),
                "anti_grounded": _patched_logits(
                    model,
                    layer,
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
            visible_prompt, condition_scores["visible"] = _visible_score(
                model, episode, demo_tasks, query_task
            )
            row: dict[str, object] = {
                "record_type": "report",
                "layer": layer,
                "fold": fold,
                "query_task": query_task.name,
                "demo_tasks": [task.name for task in demo_tasks],
                "cell_id": episode.cell_id,
                "demo_signs": list(episode.demo_signs),
                "query_sign": episode.query_sign,
                "positive_label": episode.positive_label,
                "negative_label": episode.negative_label,
                "correct_label": episode.correct_label,
                "prompt": prepared.prompt,
                "prompt_sha256": prepared.prompt_sha256,
                "visible_prompt": visible_prompt,
                "visible_prompt_sha256": sha256_text(visible_prompt),
                "token_ids": prepared.input_ids[0].tolist(),
                "state_positions": list(prepared.state_positions),
                "condition_scores": condition_scores,
            }
            rows.append(row)
            report_rows.append(row)
        print(f"report fold {fold + 1}/{len(tasks)} {query_task.name}", flush=True)
    return report_rows


def _condition_accuracy(rows: list[dict[str, object]], condition: str, key: str) -> float:
    return sum(
        bool(cast(dict[str, Any], row["condition_scores"])[condition][key]) for row in rows
    ) / len(rows)


def _paired_accuracy(rows: list[dict[str, object]], *, mapping_flip: bool = False) -> float:
    groups: dict[tuple[object, ...], list[bool]] = {}
    for row in rows:
        if mapping_flip:
            key = (row["query_task"], tuple(cast(list[int], row["demo_signs"])), row["query_sign"])
        else:
            key = (
                row["query_task"],
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
    visible = _condition_accuracy(rows, "visible", "correct_target")
    by_task: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(cast(str, row["query_task"]) for row in rows):
        subset = [row for row in rows if row["query_task"] == name]
        by_task[name] = {
            "natural_accuracy": _condition_accuracy(subset, "natural", "correct_target"),
            "query_only_accuracy": _condition_accuracy(subset, "query_only", "correct_target"),
            "visible_accuracy": _condition_accuracy(subset, "visible", "correct_target"),
        }
    positive_tasks = sum(
        values["natural_accuracy"] > values["query_only_accuracy"] for values in by_task.values()
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
        "tasks_with_positive_difference": positive_tasks,
        "by_task": by_task,
        "natural_query_twin_both_correct": _paired_accuracy(rows),
        "natural_mapping_flip_both_correct": _paired_accuracy(rows, mapping_flip=True),
        "anti_grounded_inverse_accuracy": anti_inverse,
        "visible_accuracy": visible,
        "natural_format_rate": sum(bool(score["format_ok"]) for score in natural_scores)
        / len(natural_scores),
        "natural_mean_label_mass": sum(float(score["label_mass"]) for score in natural_scores)
        / len(natural_scores),
        "sham_clean_max_label_logprob_difference": max(sham_errors),
    }
    gates = {
        "natural_accuracy": natural >= 0.75,
        "natural_minus_query_only": natural - query_only >= 0.20,
        "tasks_with_positive_difference": positive_tasks >= 4,
        "natural_query_twin_both_correct": metrics["natural_query_twin_both_correct"] >= 0.60,
        "anti_grounded_inverse_accuracy": anti_inverse >= 0.75,
        "natural_format_rate": metrics["natural_format_rate"] >= 0.90,
        "natural_mean_label_mass": metrics["natural_mean_label_mass"] >= 0.50,
        "sham_matches_clean": max(sham_errors) <= 1e-4,
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        # Not a gate on the result. It decides whether a null may be read as a
        # reporting null rather than a failure to induce the rule at all.
        "visible_capability_passed": visible >= 0.75,
    }


def _status(
    selected_layer: int | None,
    confirm: dict[str, object] | None,
    report: dict[str, object] | None,
) -> str:
    if selected_layer is None:
        return "stop_no_anchor_layer_reachable"
    if confirm is None or not confirm["passed"]:
        return "stop_test_bank_not_reachable"
    if report is None:
        return "stop_reachability_failed"
    if report["all_gates_pass"]:
        return "pilot_pass"
    return (
        "stop_reporter_gate_failed"
        if report["visible_capability_passed"]
        else "stop_reporter_gate_failed_uninterpretable"
    )


def run(args: argparse.Namespace) -> None:
    anchors = _anchors(args.site)
    out = args.out or _default_output(args.smoke, args.site)
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    protocol_path = args.protocol or _default_protocol(args.smoke, args.site)
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _freeze_protocol(protocol_path, args.smoke, args.site)
    preflight_check(MODEL, training=False)
    model = _load()
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        if models.loaded_revision(model) != MODEL_REVISION:
            raise SystemExit("loaded model revision does not match the frozen revision")
        banks = {
            bank: {
                (task.name, sign): _prepare_donor(model, task, sign, anchors)
                for task in tasks
                for sign in (1, -1)
            }
            for bank, tasks in (("dev", ARITH_DEV), ("test", ARITH_TEST))
        }
        dev_clean_ok = _clean_gate(banks["dev"], ARITH_DEV, "dev", rows)
        screen = {
            layer: _reachability(model, banks["dev"], ARITH_DEV, layer, "dev", dev_clean_ok, rows)
            for layer in anchors
        }
        for layer in anchors:
            print(f"dev layer {layer}: {json.dumps(screen[layer], sort_keys=True)}", flush=True)
        selected_layer = _select_layer(screen, anchors)

        confirm = None
        if selected_layer is not None:
            test_clean_ok = _clean_gate(banks["test"], ARITH_TEST, "test", rows)
            confirm = _reachability(
                model, banks["test"], ARITH_TEST, selected_layer, "test", test_clean_ok, rows
            )
            print(f"test bank: {json.dumps(confirm, sort_keys=True)}", flush=True)

        report_rows = (
            _report(model, banks["test"], ARITH_TEST, selected_layer, smoke=args.smoke, rows=rows)
            if selected_layer is not None and confirm is not None and confirm["passed"]
            else []
        )
        report = _report_summary(report_rows) if report_rows else None
        summary = {
            "schema_version": 2,
            "dev_layer_screen": {str(layer): result for layer, result in screen.items()},
            "selected_layer": selected_layer,
            "test_bank_reachability": confirm,
            "report": report,
            "status": _status(selected_layer, confirm, report),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out, rows)
        raw_sha = _sha256(out)
        config = {
            "schema_version": 2,
            "model": model.name,
            "model_revision": models.loaded_revision(model),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "anchor_layers": list(anchors),
            "named_site": args.site,
            "selected_layer": selected_layer,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        manifest = {
            "schema_version": 2,
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


def _default_output(smoke: bool, site: int | None = None) -> Path:
    if site is not None:
        stem = f"natural_state_arith_l{site}{'_smoke' if smoke else ''}_v1"
        return Path(f"results/{stem}_raw.jsonl")
    return Path(
        "results/natural_state_arith_smoke_v1_raw.jsonl"
        if smoke
        else "results/natural_state_arith_v1_raw.jsonl"
    )


def _default_protocol(smoke: bool, site: int | None = None) -> Path:
    if site is not None:
        marker = "_smoke" if smoke else ""
        return Path(f"results/natural_state_arith_l{site}{marker}_protocol_v1.json")
    return Path(
        "results/natural_state_arith_smoke_protocol_v1.json"
        if smoke
        else "results/natural_state_arith_protocol_v1.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--site",
        type=int,
        help="run one named layer instead of the blind anchor screen; the protocol "
        "records it as a post-hoc selection",
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
