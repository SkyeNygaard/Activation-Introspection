"""DEV-only layer localization for causal hidden-state codebook reporting.

This is a discovery instrument, not confirmation evidence.  It replaces every
query-head context at one downstream layer with the exact paired ``test_only``
donor, separately at four preregistered receiver roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
# These must precede the imports that transitively load Hugging Face.
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(ROOT / "src"))

from introspect import concepts, models, retained  # noqa: E402
from introspect.attention_patching import (  # noqa: E402
    capture_attention_inputs,
    patch_attention_inputs,
)
from introspect.codebook_icl import (  # noqa: E402
    LABELS,
    VISIBLE_SAMPLES,
    Condition,
    Episode,
    PreparedEpisode,
    condition_interventions,
    exact_episodes,
    prepare_episode,
    sha256_text,
    tensor_sha256,
)
from introspect.concepts import ConceptVector  # noqa: E402
from introspect.hooks import intervene  # noqa: E402
from introspect.models import LoadedModel  # noqa: E402

MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CONCEPT = "ocean"
CARRIER = VISIBLE_SAMPLES[0]
INJECTION_LAYER = 9
SENTINEL_LAYERS = tuple(range(7, 10))
DOWNSTREAM_LAYERS = tuple(range(10, 36))
STRENGTH = 1.0
SHAM_LOGIT_TOLERANCE = 1e-6
RECEIVER_ROLES = ("demo_labels", "query_marker", "final_answer", "all_positions")
ReceiverPositions = Literal["all"] | tuple[int, ...]

_LABEL_RE = re.compile(r"Label: ([QK])")
_SOURCE_PATHS = (
    "src/introspect/attention_patching.py",
    "src/introspect/codebook_icl.py",
    "src/introspect/concepts.py",
    "src/introspect/hooks.py",
    "src/introspect/models.py",
    "src/introspect/retained.py",
    "scripts/run_attention_localization.py",
    "pyproject.toml",
    "uv.lock",
)


def balanced_dev_episodes(sample: str) -> list[Episode]:
    """All six demo orders, alternating mappings, with both query-sign twins."""
    episodes = exact_episodes(sample)
    orders = sorted({episode.demo_signs for episode in episodes})
    selected = [
        episode
        for order_id, order in enumerate(orders)
        for episode in episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[order_id % 2]
    ]
    if (
        len(selected) != 12
        or len({episode.demo_signs for episode in selected}) != 6
        or sum(episode.query_sign == 1 for episode in selected) != 6
        or sum(episode.positive_label == "Q" for episode in selected) != 6
        or sum(episode.correct_label == "Q" for episode in selected) != 6
    ):
        raise RuntimeError("the DEV subset lost order, mapping, query-sign, or label balance")
    return selected


def _demo_label_positions(model: LoadedModel, prepared: PreparedEpisode) -> tuple[int, ...]:
    matches = list(_LABEL_RE.finditer(prepared.prompt))
    expected_labels = [prepared.episode.label_for(sign) for sign in prepared.episode.demo_signs]
    if len(matches) != 4 or [match.group(1) for match in matches] != expected_labels:
        raise ValueError("could not identify the four demonstration-label spans exactly")

    encoded = cast(
        Any,
        model.tokenizer(prepared.prompt, return_tensors="pt", return_offsets_mapping=True),
    )
    input_ids = cast(Tensor, encoded.input_ids)
    if not torch.equal(input_ids.to(prepared.input_ids.device), prepared.input_ids):
        raise ValueError("offset tokenization changed the prepared input IDs")
    offsets = cast(Tensor, encoded.offset_mapping)[0].tolist()

    positions: list[int] = []
    ids_by_label = dict(zip(LABELS, prepared.label_ids, strict=True))
    for match in matches:
        start, end = match.span(1)
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if len(overlapping) != 1:
            raise ValueError("each demonstration label must map to exactly one token")
        position = overlapping[0]
        if int(input_ids[0, position]) != ids_by_label[match.group(1)]:
            raise ValueError("demonstration and answer label token IDs differ")
        positions.append(position)
    if len(set(positions)) != 4:
        raise ValueError("demonstration labels mapped to duplicate token positions")
    return tuple(positions)


def receiver_roles(
    prepared: PreparedEpisode, demo_labels: tuple[int, ...]
) -> dict[str, ReceiverPositions]:
    query_marker = prepared.state_positions[-1]
    answer = int(prepared.input_ids.shape[1]) - 1
    if not (
        len(demo_labels) == 4
        and all(
            marker < label
            for marker, label in zip(prepared.state_positions[:4], demo_labels, strict=True)
        )
        and max(demo_labels) < query_marker < answer
    ):
        raise ValueError("receiver positions violate the frozen causal ordering")
    return {
        "demo_labels": demo_labels,
        "query_marker": (query_marker,),
        "final_answer": (answer,),
        "all_positions": "all",
    }


def _score(logits: Tensor, prepared: PreparedEpisode) -> dict[str, object]:
    if logits.ndim != 1:
        raise ValueError("expected one vocabulary-logit vector")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    candidates = torch.tensor(prepared.label_ids, device=logits.device)
    selected = logits.float()[candidates]
    conditional = torch.softmax(selected, dim=0)
    full = torch.log_softmax(logits.float(), dim=0)
    correct_index = LABELS.index(prepared.episode.correct_label)
    other_index = 1 - correct_index
    predicted = LABELS[int(selected.argmax())]
    return {
        "predicted_label": predicted,
        "correct": predicted == prepared.episode.correct_label,
        "signed_margin": float(selected[correct_index] - selected[other_index]),
        "conditional_correct_probability": float(conditional[correct_index]),
        "label_logits": {label: float(selected[index]) for index, label in enumerate(LABELS)},
        "full_logprobs": {
            label: float(full[token_id])
            for label, token_id in zip(LABELS, prepared.label_ids, strict=True)
        },
        "label_mass": float(
            torch.logsumexp(selected, 0).sub(torch.logsumexp(logits.float(), 0)).exp()
        ),
        "format_ok": int(logits.argmax()) in set(prepared.label_ids),
        "full_logits_sha256": tensor_sha256(logits),
    }


def _kl_nats(reference: Tensor, comparison: Tensor) -> float:
    reference_logprobs = torch.log_softmax(reference.float(), dim=0)
    comparison_logprobs = torch.log_softmax(comparison.float(), dim=0)
    return float((reference_logprobs.exp() * (reference_logprobs - comparison_logprobs)).sum())


def _forward_logits(model: LoadedModel, input_ids: Tensor) -> Tensor:
    return cast(Tensor, cast(Any, model.model)(input_ids, use_cache=False).logits)


@torch.no_grad()
def _capture_condition(
    model: LoadedModel,
    prepared: PreparedEpisode,
    condition: Condition,
    direction: ConceptVector,
    layers: tuple[int, ...],
) -> tuple[dict[str, object], Tensor, dict[int, Tensor]]:
    interventions = condition_interventions(
        condition,
        direction,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=STRENGTH,
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        capture_attention_inputs(model, layers) as captured,
    ):
        logits = _forward_logits(model, prepared.input_ids)[0, -1].float()
    if set(captured.by_layer) != set(layers):
        raise RuntimeError("attention capture missed a requested layer")
    return _score(logits, prepared), logits.detach().cpu(), captured.by_layer


@torch.no_grad()
def _patched_score(
    model: LoadedModel,
    prepared: PreparedEpisode,
    direction: ConceptVector,
    *,
    layer: int,
    role: ReceiverPositions,
    n_heads: int,
    donor: Tensor,
    expected_recipient: Tensor,
    target_logits: Tensor,
) -> dict[str, object]:
    interventions = condition_interventions(
        "target",
        direction,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=STRENGTH,
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        patch_attention_inputs(
            model,
            {layer: donor},
            {layer: range(n_heads)},
            n_heads=n_heads,
            expected_recipients={layer: expected_recipient},
            positions=role,
        ),
    ):
        logits = _forward_logits(model, prepared.input_ids)[0, -1].float()
    score = _score(logits, prepared)
    score["target_to_patched_kl_nats"] = _kl_nats(target_logits, logits)
    return score


@torch.no_grad()
def _self_patch(
    model: LoadedModel,
    prepared: PreparedEpisode,
    direction: ConceptVector,
    *,
    layers: tuple[int, ...],
    n_heads: int,
    target_contexts: dict[int, Tensor],
) -> tuple[dict[str, object], Tensor]:
    interventions = condition_interventions(
        "target",
        direction,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=STRENGTH,
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        patch_attention_inputs(
            model,
            target_contexts,
            {layer: range(n_heads) for layer in layers},
            n_heads=n_heads,
            expected_recipients=target_contexts,
            positions="all",
        ),
    ):
        logits = _forward_logits(model, prepared.input_ids)[0, -1].float()
    return _score(logits, prepared), logits.detach().cpu()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256_text(raw)


def _source_hashes(root: Path) -> dict[str, str]:
    return {path: _sha256(root / path) for path in _SOURCE_PATHS}


def _expected_protocol(source_hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "frozen_on": "2026-08-10",
        "status": "DEV_ONLY_LAYER_SCREEN_NOT_CONFIRMATORY",
        "safety_goal": (
            "Test whether demonstration-mediated hidden-state reporting has a small, "
            "causally auditable downstream attention route before attempting a "
            "shortcut-resistant programmatic reporter."
        ),
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
            "concept": CONCEPT,
            "carrier": CARRIER,
            "injection_layer": INJECTION_LAYER,
            "strength": STRENGTH,
            "pre_injection_sentinel_layers": list(SENTINEL_LAYERS),
            "downstream_layers": list(DOWNSTREAM_LAYERS),
            "receiver_roles": list(RECEIVER_ROLES),
            "n_query_heads": 16,
            "n_kv_heads": 2,
            "head_width": 128,
            "cells": (
                "all six balanced demonstration orders; mapping alternates by order; "
                "both query-sign twins (12 cells)"
            ),
            "interchange": (
                "replace all 16 pre-o_proj query-head contexts at one layer/receiver role "
                "with the exact paired test_only context"
            ),
        },
        "instrumentation_gates": {
            "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
            "model_loading": "pinned revision from the project-local cache in offline mode",
            "pre_injection_sentinels": "target and test_only must be bit-identical",
            "self_donor": (
                f"all-layer/all-position target self-patch max full-logit error <= "
                f"{SHAM_LOGIT_TOLERANCE}"
            ),
            "recipient_binding": "each patched pre-o_proj input must equal its cached target",
            "artifact_binding": "save prompt, token, direction, donor, recipient, and logit hashes",
        },
        "analysis_rules": {
            "unit": (
                "12 exact nuisance cells for one concept/carrier family; descriptive "
                "development screening only"
            ),
            "primary": (
                "signed correct-minus-wrong Q/K margin; report conditional correct "
                "probability, accuracy, format, label mass, and target-to-patch KL"
            ),
            "removal_fraction": (
                "ratio of aggregate paired mean margin drops: "
                "mean(target-patch)/mean(target-test_only); never average rowwise ratios"
            ),
            "baseline_gate": (
                "target-test_only conditional-correct-probability gap >= 0.15 and accuracy "
                "gap >= 0.25"
            ),
            "candidate_gate": (
                "a syntax-specific role removes >= 20% of the aggregate baseline margin, "
                "retains format rate >= 0.90, and retains mean label mass >= 80% of target"
            ),
            "selection": (
                "carry at most the top two layers per syntax-specific role to a separately "
                "frozen multi-concept/head screen; all_positions is diagnostic only"
            ),
            "stop": (
                "stop this route grammar if the baseline gate fails, no syntax-specific role "
                "passes, or only the all_positions envelope passes; do not inspect individual "
                "heads under this protocol"
            ),
            "overlap": "receiver-role effects overlap and are not additive",
            "inference": "no p-values or population intervals from this single-family screen",
        },
        "claim_boundary": (
            "A positive result is layer/receiver-role sensitivity under a paired hybrid "
            "interchange, not QK-only mediation, natural indirect effect, semantic "
            "introspection, safety-monitor robustness, or a readable program. A null does not "
            "exclude distributed, redundant, residual, MLP-carried, or compensatory routes."
        ),
        "smoke_disclosure": (
            "Before the full screen, one query-twin pair and layer 10 may be inspected only "
            "for instrumentation. Any source/design change requires a new protocol; smoke "
            "outcomes may not select layers or alter gates."
        ),
        "previous_attempt_disclosure": (
            "Protocol v1 was frozen, then its first smoke launch failed before model loading "
            "because the project-local Hugging Face cache was not selected and network access "
            "was unavailable. No model output or result artifact existed. V2 changes only "
            "offline cache selection and records that execution mode."
        ),
        "source_files_sha256": source_hashes,
    }


def _load_protocol(path: Path, source_hashes: dict[str, str]) -> tuple[dict[str, object], str]:
    raw = path.read_text()
    protocol = cast(dict[str, object], json.loads(raw))
    if protocol != _expected_protocol(source_hashes):
        raise RuntimeError("protocol does not match the frozen executable design and sources")
    return protocol, sha256_text(raw)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty(root: Path) -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:
        return True


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary file: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(*, out: Path, smoke: bool, protocol_path: Path) -> None:
    root = ROOT
    manifest_path = out.with_suffix(".manifest.json")
    temporary = out.with_name(f".{out.name}.tmp")
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if out.exists() or manifest_path.exists() or temporary.exists() or manifest_temporary.exists():
        raise SystemExit("refusing to overwrite an existing raw, manifest, or temporary artifact")

    source_hashes = _source_hashes(root)
    protocol, protocol_sha = _load_protocol(protocol_path, source_hashes)
    layers = DOWNSTREAM_LAYERS[:1] if smoke else DOWNSTREAM_LAYERS
    episodes = balanced_dev_episodes(CARRIER)
    if smoke:
        episodes = episodes[:2]

    torch.manual_seed(0)
    model = models.load(MODEL, device=torch.device("cpu"), revision=MODEL_REVISION)
    try:
        if model.n_layers != 36 or layers[0] != INJECTION_LAYER + 1 or layers[-1] >= model.n_layers:
            raise RuntimeError("the loaded model does not match the frozen 36-layer design")
        loaded_revision = models.loaded_revision(model)
        if loaded_revision != MODEL_REVISION:
            raise RuntimeError(f"loaded revision {loaded_revision} does not match {MODEL_REVISION}")
        if model.device != torch.device("cpu") or model.dtype != torch.float32:
            raise RuntimeError("localization is frozen to CPU float32")
        if cast(Any, model.model).training:
            raise RuntimeError("localization requires eval mode with dropout disabled")
        config_object = cast(Any, model.model).config
        n_heads = int(config_object.num_attention_heads)
        n_kv_heads = int(config_object.num_key_value_heads)
        if n_heads != 16 or n_kv_heads != 2 or model.d_model != 2048:
            raise RuntimeError("loaded Qwen head layout differs from the frozen 16Q/2KV design")

        raw_bank = concepts.build_bank(
            model, INJECTION_LAYER, list(retained.DEV_CONCEPTS), center=False
        )
        center = torch.stack([item.vector for item in raw_bank.values()]).mean(0)
        bank = {
            name: ConceptVector(name, INJECTION_LAYER, item.vector - center)
            for name, item in raw_bank.items()
        }
        max_cosine = concepts.max_offdiagonal_cosine(bank)
        if max_cosine > 0.5:
            raise RuntimeError(f"centered DEV directions are near-collinear ({max_cosine:.3f})")
        direction = bank[CONCEPT]

        prepared = [prepare_episode(model, episode) for episode in episodes]
        labels_by_cell = {
            item.episode.cell_id: _demo_label_positions(model, item) for item in prepared
        }
        roles_by_cell = {
            item.episode.cell_id: receiver_roles(item, labels_by_cell[item.episode.cell_id])
            for item in prepared
        }
        prompt_hashes = [item.prompt_sha256 for item in prepared]
        config: dict[str, object] = {
            "schema_version": 1,
            "status": "DEV_ONLY_NOT_CONFIRMATORY",
            "estimand": "layerwise necessity for demonstration-mediated hidden-state reporting",
            "model_requested": MODEL,
            "model_resolved": model.name,
            "model_revision": loaded_revision,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "concept": CONCEPT,
            "carrier": CARRIER,
            "injection_layer": INJECTION_LAYER,
            "strength": STRENGTH,
            "layers": list(layers),
            "pre_injection_sentinel_layers": list(SENTINEL_LAYERS),
            "receiver_roles": list(RECEIVER_ROLES),
            "patch": "all query-head contexts from the paired test_only donor before o_proj",
            "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
            "hf_home": os.environ["HF_HOME"],
            "offline_model_loading": True,
            "self_patch_full_logit_tolerance": SHAM_LOGIT_TOLERANCE,
            "n_query_heads": n_heads,
            "head_width": model.d_model // n_heads,
            "balanced_subset": "six demo orders; alternating mapping; both query-sign twins",
            "cell_ids": [item.episode.cell_id for item in prepared],
            "prompt_set_sha256": _json_sha256(sorted(prompt_hashes)),
            "direction_sha256": tensor_sha256(direction.vector),
            "centering_direction_sha256": tensor_sha256(center),
            "centering_concepts": list(retained.DEV_CONCEPTS),
            "max_offdiagonal_cosine": max_cosine,
            "source_files_sha256": source_hashes,
            "source_sha256": _json_sha256(source_hashes),
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "smoke": smoke,
            "publishable": False,
            "git_commit": _git_commit(root),
            "git_dirty": _git_dirty(root),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        config_sha = _json_sha256(config)
        started = time.time()
        rows = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("x") as handle:
                for index, item in enumerate(prepared, start=1):
                    capture_layers = SENTINEL_LAYERS + layers
                    target_score, target_logits, target_contexts = _capture_condition(
                        model, item, "target", direction, capture_layers
                    )
                    target_score["target_to_target_kl_nats"] = 0.0
                    test_only_score, test_only_logits, donor_contexts = _capture_condition(
                        model, item, "test_only", direction, capture_layers
                    )
                    test_only_score["target_to_test_only_kl_nats"] = _kl_nats(
                        target_logits, test_only_logits
                    )
                    for layer in SENTINEL_LAYERS:
                        if not torch.equal(target_contexts[layer], donor_contexts[layer]):
                            raise RuntimeError(
                                "target/test_only diverged before the intervention at "
                                f"layer {layer}"
                            )
                    self_patch_score, self_patch_logits = _self_patch(
                        model,
                        item,
                        direction,
                        layers=layers,
                        n_heads=n_heads,
                        target_contexts={layer: target_contexts[layer] for layer in layers},
                    )
                    self_patch_error = float((target_logits - self_patch_logits).abs().max())
                    if self_patch_error > SHAM_LOGIT_TOLERANCE:
                        raise RuntimeError(
                            f"self-donor patch changed logits by {self_patch_error:.3g}"
                        )
                    self_patch_score["target_to_self_patch_kl_nats"] = _kl_nats(
                        target_logits, self_patch_logits
                    )
                    patched: dict[str, dict[str, object]] = {}
                    roles = roles_by_cell[item.episode.cell_id]
                    for layer in layers:
                        patched[str(layer)] = {
                            role_name: _patched_score(
                                model,
                                item,
                                direction,
                                layer=layer,
                                role=positions,
                                n_heads=n_heads,
                                donor=donor_contexts[layer],
                                expected_recipient=target_contexts[layer],
                                target_logits=target_logits,
                            )
                            for role_name, positions in roles.items()
                        }

                    token_ids = item.input_ids[0].tolist()
                    row = {
                        "schema_version": 1,
                        "config_sha256": config_sha,
                        "cell_id": item.episode.cell_id,
                        "episode_sha256": item.episode.digest(),
                        "prompt": item.prompt,
                        "prompt_sha256": item.prompt_sha256,
                        "token_ids_sha256": _json_sha256(token_ids),
                        "token_ids": token_ids,
                        "state_token_positions": item.state_positions,
                        "demo_label_positions": labels_by_cell[item.episode.cell_id],
                        "final_answer_position": int(item.input_ids.shape[1]) - 1,
                        "demo_signs": item.episode.demo_signs,
                        "query_sign": item.episode.query_sign,
                        "label_mapping": {
                            "+1": item.episode.positive_label,
                            "-1": item.episode.negative_label,
                        },
                        "correct_label": item.episode.correct_label,
                        "baseline": {"target": target_score, "test_only": test_only_score},
                        "instrumentation": {
                            "pre_injection_sentinels_equal": True,
                            "self_patch_score": self_patch_score,
                            "self_patch_max_abs_logit_error": self_patch_error,
                            "attention_context_sha256": {
                                "target": {
                                    str(layer): tensor_sha256(target_contexts[layer])
                                    for layer in capture_layers
                                },
                                "test_only": {
                                    str(layer): tensor_sha256(donor_contexts[layer])
                                    for layer in capture_layers
                                },
                            },
                        },
                        "patched": patched,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    rows += 1
                    print(
                        f"cell {index}/{len(prepared)} {item.episode.cell_id} "
                        f"[{time.time() - started:.0f}s]",
                        flush=True,
                    )
            if source_hashes != _source_hashes(root):
                raise RuntimeError("generation source changed while the run was active")
            temporary.replace(out)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        raw_sha = _sha256(out)
        n_forwards = rows * (3 + len(layers) * len(config["receiver_roles"]))  # type: ignore[arg-type]
        manifest = {
            "schema_version": 1,
            "config": config,
            "config_sha256": config_sha,
            "raw": out.name,
            "raw_sha256": raw_sha,
            "n_episode_rows": rows,
            "n_scored_forwards": n_forwards,
        }
        _write_json(manifest_path, manifest)
        print(f"wrote {out} (sha256={raw_sha}, scored_forwards={n_forwards})")
        print(f"manifest {manifest_path}")
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/attention_localization_dev_protocol_v2.json"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out or Path(
        "results/attention_localization_smoke_raw.jsonl"
        if args.smoke
        else "results/attention_localization_dev_raw.jsonl"
    )
    run(out=out, smoke=args.smoke, protocol_path=args.protocol)


if __name__ == "__main__":
    main()
