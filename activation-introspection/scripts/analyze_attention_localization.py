"""Fail-closed analysis for the frozen DEV-only attention layer screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect.codebook_icl import LABELS, Episode, exact_episodes, sha256_text

PROTOCOL_SHA256 = "27c8af5f4917dc3c72214caece71ac996d9c26ec914dd390f86f954a73e41427"
LAYERS = tuple(range(10, 36))
SENTINEL_LAYERS = (7, 8, 9)
ROLES = ("demo_labels", "query_marker", "final_answer", "all_positions")
SYNTAX_ROLES = ROLES[:-1]
BASELINE_PROBABILITY_GAP = 0.15
BASELINE_ACCURACY_GAP = 0.25
CANDIDATE_REMOVAL_FRACTION = 0.20
MIN_FORMAT_RATE = 0.90
MIN_LABEL_MASS_RETENTION = 0.80
SELF_PATCH_TOLERANCE = 1e-6
_FLOAT_TOLERANCE = 2e-5
_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} is not a SHA-256 digest")
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} is outside its valid range")
    return number


def _probability(value: object, name: str) -> float:
    number = _number(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} is not a probability")
    return number


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    digest = _sha256(path)
    if digest != PROTOCOL_SHA256:
        raise ValueError(f"protocol SHA-256 drifted: {digest}")
    protocol = cast(dict[str, Any], json.loads(path.read_text()))
    sources = cast(dict[str, str], protocol["source_files_sha256"])
    root = Path(__file__).resolve().parents[1]
    if any(_sha256(root / source) != expected for source, expected in sources.items()):
        raise ValueError("a protocol-bound generation source drifted")
    return protocol, digest


def load_verified(
    raw_path: Path, manifest_path: Path, protocol_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    protocol, protocol_sha = load_protocol(protocol_path)
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    config = cast(dict[str, Any], manifest["config"])
    if _json_sha256(config) != manifest.get("config_sha256"):
        raise ValueError("manifest config SHA-256 does not match its config")
    if config.get("protocol") != protocol or config.get("protocol_sha256") != protocol_sha:
        raise ValueError("manifest config is not bound to the frozen protocol")
    if config.get("source_files_sha256") != protocol["source_files_sha256"]:
        raise ValueError("manifest source hashes differ from the protocol")
    if _json_sha256(config["source_files_sha256"]) != config.get("source_sha256"):
        raise ValueError("manifest aggregate source hash drifted")
    raw_sha = _sha256(raw_path)
    if raw_sha != manifest.get("raw_sha256") or manifest.get("raw") != raw_path.name:
        raise ValueError("raw artifact SHA-256 or filename does not match its manifest")
    rows = cast(
        list[dict[str, Any]],
        [json.loads(line) for line in raw_path.read_text().splitlines() if line],
    )
    if len(rows) != manifest.get("n_episode_rows"):
        raise ValueError("raw row count does not match its manifest")
    if any(row.get("config_sha256") != manifest["config_sha256"] for row in rows):
        raise ValueError("a raw row refers to a different config")
    return rows, manifest, protocol, protocol_sha


def _balanced_episodes(carrier: str) -> list[Episode]:
    episodes = exact_episodes(carrier)
    orders = sorted({episode.demo_signs for episode in episodes})
    return [
        episode
        for order_id, order in enumerate(orders)
        for episode in episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[order_id % 2]
    ]


def _validate_score(
    score: dict[str, Any], *, correct_label: str, kl_key: str, location: str
) -> None:
    if not isinstance(score.get("correct"), bool) or not isinstance(score.get("format_ok"), bool):
        raise ValueError(f"non-boolean score flag in {location}")
    logits = score.get("label_logits")
    logprobs = score.get("full_logprobs")
    if not isinstance(logits, dict) or set(logits) != set(LABELS):
        raise ValueError(f"label-logit keys drifted in {location}")
    if not isinstance(logprobs, dict) or set(logprobs) != set(LABELS):
        raise ValueError(f"label-logprob keys drifted in {location}")
    label_logits = {label: _number(logits[label], f"{location}/{label}-logit") for label in LABELS}
    label_logprobs = {
        label: _number(logprobs[label], f"{location}/{label}-logprob") for label in LABELS
    }
    if any(value > _FLOAT_TOLERANCE for value in label_logprobs.values()):
        raise ValueError(f"positive full-vocabulary log-probability in {location}")

    correct_index = LABELS.index(correct_label)
    other_label = LABELS[1 - correct_index]
    expected_margin = label_logits[correct_label] - label_logits[other_label]
    margin = _number(score.get("signed_margin"), f"{location}/margin")
    if not math.isclose(margin, expected_margin, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"signed margin is inconsistent in {location}")
    expected_probability = 1 / (1 + math.exp(-max(-700.0, min(700.0, margin))))
    probability = _probability(
        score.get("conditional_correct_probability"), f"{location}/correct-probability"
    )
    if not math.isclose(probability, expected_probability, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"conditional probability is inconsistent in {location}")

    masses = {label: math.exp(label_logprobs[label]) for label in LABELS}
    derived_mass = sum(masses.values())
    if derived_mass <= 0:
        raise ValueError(f"label mass underflowed in {location}")
    label_mass = _probability(score.get("label_mass"), f"{location}/label-mass")
    if not math.isclose(label_mass, derived_mass, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"label mass is inconsistent in {location}")
    derived_probability = masses[correct_label] / derived_mass
    if not math.isclose(probability, derived_probability, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"label logits and full log-probabilities disagree in {location}")
    if score["format_ok"] and max(masses.values()) <= max(0.0, 1 - derived_mass) + _FLOAT_TOLERANCE:
        raise ValueError(
            f"format_ok=True cannot be certified from saved probabilities in {location}"
        )

    predicted = max(LABELS, key=label_logits.__getitem__)
    if score.get("predicted_label") != predicted:
        raise ValueError(f"predicted label is inconsistent in {location}")
    if score["correct"] != (predicted == correct_label):
        raise ValueError(f"correctness flag is inconsistent in {location}")
    _hash(score.get("full_logits_sha256"), f"{location}/full-logits")
    if kl_key not in score or _number(score[kl_key], f"{location}/{kl_key}") < -1e-6:
        raise ValueError(f"invalid KL divergence in {location}")


def _validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    design = protocol["design"]
    expected = {
        "schema_version": 1,
        "status": "DEV_ONLY_NOT_CONFIRMATORY",
        "estimand": "layerwise necessity for demonstration-mediated hidden-state reporting",
        "model_requested": design["model"],
        "model_resolved": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": design["model_revision"],
        "device": design["device"],
        "dtype": "torch.float32",
        "concept": design["concept"],
        "carrier": design["carrier"],
        "injection_layer": design["injection_layer"],
        "strength": design["strength"],
        "layers": list(LAYERS),
        "pre_injection_sentinel_layers": list(SENTINEL_LAYERS),
        "receiver_roles": list(ROLES),
        "patch": "all query-head contexts from the paired test_only donor before o_proj",
        "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
        "offline_model_loading": True,
        "self_patch_full_logit_tolerance": SELF_PATCH_TOLERANCE,
        "n_query_heads": design["n_query_heads"],
        "head_width": design["head_width"],
        "smoke": False,
        "publishable": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"full-run config field {key!r} drifted")
    if config.get("cell_ids") != [
        episode.cell_id for episode in _balanced_episodes(design["carrier"])
    ]:
        raise ValueError("configured 12-cell subset drifted")
    for key in ("direction_sha256", "centering_direction_sha256", "prompt_set_sha256"):
        _hash(config.get(key), f"config/{key}")
    max_cosine = _number(config.get("max_offdiagonal_cosine"), "config/max-cosine", minimum=0)
    if max_cosine > 0.5:
        raise ValueError("configured DEV directions violate the cosine gate")


def validate_design(
    rows: list[dict[str, Any]], manifest: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, int]:
    config = cast(dict[str, Any], manifest["config"])
    _validate_config(config, protocol)
    if manifest.get("schema_version") != 1 or manifest.get("n_episode_rows") != 12:
        raise ValueError("manifest schema or row count drifted")
    expected_forwards = 12 * (3 + len(LAYERS) * len(ROLES))
    if manifest.get("n_scored_forwards") != expected_forwards:
        raise ValueError("manifest scored-forward count drifted")

    episodes = {episode.cell_id: episode for episode in _balanced_episodes(config["carrier"])}
    if len(rows) != 12 or {row.get("cell_id") for row in rows} != set(episodes):
        raise ValueError("the exact 12-cell DEV subset is incomplete")
    prompt_hashes: list[str] = []
    twin_prompts: dict[str, set[str]] = {}
    patch_count = 0
    for row in rows:
        cell_id = cast(str, row["cell_id"])
        episode = episodes[cell_id]
        location = f"cell/{cell_id}"
        if row.get("schema_version") != 1:
            raise ValueError(f"row schema drifted in {location}")
        expected_mapping = {"+1": episode.positive_label, "-1": episode.negative_label}
        if (
            row.get("episode_sha256") != episode.digest()
            or tuple(row.get("demo_signs", ())) != episode.demo_signs
            or row.get("query_sign") != episode.query_sign
            or row.get("label_mapping") != expected_mapping
            or row.get("correct_label") != episode.correct_label
        ):
            raise ValueError(f"episode fields drifted in {location}")

        prompt = row.get("prompt")
        if not isinstance(prompt, str) or sha256_text(prompt) != row.get("prompt_sha256"):
            raise ValueError(f"prompt hash drifted in {location}")
        if prompt.count(episode.render_user()) != 1:
            raise ValueError(f"prompt does not contain the exact episode in {location}")
        prompt_hashes.append(row["prompt_sha256"])
        twin_prompts.setdefault(cell_id.rsplit("q", 1)[0], set()).add(prompt)

        token_ids = row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
            or _json_sha256(token_ids) != row.get("token_ids_sha256")
        ):
            raise ValueError(f"token IDs drifted in {location}")
        state_positions = row.get("state_token_positions")
        demo_labels = row.get("demo_label_positions")
        answer = row.get("final_answer_position")
        if (
            not isinstance(state_positions, list)
            or len(state_positions) != 5
            or any(
                not isinstance(position, int) or isinstance(position, bool) or position < 0
                for position in state_positions
            )
            or state_positions != sorted(set(state_positions))
            or not isinstance(demo_labels, list)
            or len(demo_labels) != 4
            or any(
                not isinstance(position, int) or isinstance(position, bool) or position < 0
                for position in demo_labels
            )
            or demo_labels != sorted(set(demo_labels))
            or not all(
                marker < label
                for marker, label in zip(state_positions[:4], demo_labels, strict=True)
            )
            or not isinstance(answer, int)
            or isinstance(answer, bool)
            or not max(demo_labels) < state_positions[-1] < answer == len(token_ids) - 1
        ):
            raise ValueError(f"receiver positions drifted in {location}")

        baseline = row.get("baseline")
        if not isinstance(baseline, dict) or set(baseline) != {"target", "test_only"}:
            raise ValueError(f"baseline arms drifted in {location}")
        target = cast(dict[str, Any], baseline["target"])
        test_only = cast(dict[str, Any], baseline["test_only"])
        _validate_score(
            target,
            correct_label=episode.correct_label,
            kl_key="target_to_target_kl_nats",
            location=f"{location}/target",
        )
        _validate_score(
            test_only,
            correct_label=episode.correct_label,
            kl_key="target_to_test_only_kl_nats",
            location=f"{location}/test_only",
        )
        if abs(float(target["target_to_target_kl_nats"])) > _FLOAT_TOLERANCE:
            raise ValueError(f"target self-KL is nonzero in {location}")

        instrumentation = row.get("instrumentation")
        if not isinstance(instrumentation, dict):
            raise ValueError(f"instrumentation record missing in {location}")
        error = _number(
            instrumentation.get("self_patch_max_abs_logit_error"),
            f"{location}/self-patch-error",
            minimum=0,
        )
        if (
            instrumentation.get("pre_injection_sentinels_equal") is not True
            or error > SELF_PATCH_TOLERANCE
        ):
            raise ValueError(f"instrumentation gate failed in {location}")
        self_patch = cast(dict[str, Any], instrumentation.get("self_patch_score"))
        _validate_score(
            self_patch,
            correct_label=episode.correct_label,
            kl_key="target_to_self_patch_kl_nats",
            location=f"{location}/self-patch",
        )
        if float(self_patch["target_to_self_patch_kl_nats"]) > _FLOAT_TOLERANCE:
            raise ValueError(f"self-patch KL exceeds tolerance in {location}")
        if error != 0.0 or self_patch["full_logits_sha256"] != target["full_logits_sha256"]:
            raise ValueError(
                f"self-patch full logits are not exactly bound to target in {location}"
            )
        if any(
            not math.isclose(float(self_patch[key]), float(target[key]), abs_tol=_FLOAT_TOLERANCE)
            for key in ("signed_margin", "conditional_correct_probability", "label_mass")
        ) or any(
            self_patch[key] != target[key] for key in ("predicted_label", "correct", "format_ok")
        ):
            raise ValueError(f"self patch changed scored behavior in {location}")

        contexts = instrumentation.get("attention_context_sha256")
        expected_context_layers = {str(layer) for layer in SENTINEL_LAYERS + LAYERS}
        if not isinstance(contexts, dict) or set(contexts) != {"target", "test_only"}:
            raise ValueError(f"attention-context hashes missing in {location}")
        for condition in ("target", "test_only"):
            condition_hashes = contexts[condition]
            if (
                not isinstance(condition_hashes, dict)
                or set(condition_hashes) != expected_context_layers
            ):
                raise ValueError(f"attention-context layer grid drifted in {location}/{condition}")
            for layer, digest in condition_hashes.items():
                _hash(digest, f"{location}/{condition}/{layer}")
        if any(
            contexts["target"][str(layer)] != contexts["test_only"][str(layer)]
            for layer in SENTINEL_LAYERS
        ):
            raise ValueError(f"pre-injection hashes differ in {location}")

        patched = row.get("patched")
        if not isinstance(patched, dict) or set(patched) != {str(layer) for layer in LAYERS}:
            raise ValueError(f"patched layer grid drifted in {location}")
        for layer in LAYERS:
            role_scores = patched[str(layer)]
            if not isinstance(role_scores, dict) or set(role_scores) != set(ROLES):
                raise ValueError(f"patched receiver-role grid drifted in {location}/layer-{layer}")
            for role in ROLES:
                _validate_score(
                    role_scores[role],
                    correct_label=episode.correct_label,
                    kl_key="target_to_patched_kl_nats",
                    location=f"{location}/layer-{layer}/{role}",
                )
                patch_count += 1

    if len(twin_prompts) != 6 or any(len(prompts) != 1 for prompts in twin_prompts.values()):
        raise ValueError("byte-identical query twins are incomplete")
    if _json_sha256(sorted(prompt_hashes)) != config["prompt_set_sha256"]:
        raise ValueError("prompt-set SHA-256 drifted")
    if patch_count != 12 * len(LAYERS) * len(ROLES):
        raise ValueError("12 x 26 x 4 patch grid is incomplete")
    return {"cells": 12, "layers": len(LAYERS), "roles": len(ROLES), "patches": patch_count}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _aggregate(scores: list[dict[str, Any]], kl_key: str) -> dict[str, float]:
    return {
        "mean_signed_margin": _mean([float(score["signed_margin"]) for score in scores]),
        "mean_conditional_correct_probability": _mean(
            [float(score["conditional_correct_probability"]) for score in scores]
        ),
        "accuracy": _mean([float(score["correct"]) for score in scores]),
        "format_rate": _mean([float(score["format_ok"]) for score in scores]),
        "mean_label_mass": _mean([float(score["label_mass"]) for score in scores]),
        "mean_kl_nats": _mean([float(score[kl_key]) for score in scores]),
    }


def analyse(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    design = validate_design(rows, manifest, protocol)
    target_scores = [cast(dict[str, Any], row["baseline"]["target"]) for row in rows]
    test_only_scores = [cast(dict[str, Any], row["baseline"]["test_only"]) for row in rows]
    target = _aggregate(target_scores, "target_to_target_kl_nats")
    test_only = _aggregate(test_only_scores, "target_to_test_only_kl_nats")
    margin_gap = target["mean_signed_margin"] - test_only["mean_signed_margin"]
    probability_gap = (
        target["mean_conditional_correct_probability"]
        - test_only["mean_conditional_correct_probability"]
    )
    accuracy_gap = target["accuracy"] - test_only["accuracy"]
    baseline_passes = (
        probability_gap >= BASELINE_PROBABILITY_GAP and accuracy_gap >= BASELINE_ACCURACY_GAP
    )

    candidates: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in ROLES}
    for role in ROLES:
        for layer in LAYERS:
            scores = [cast(dict[str, Any], row["patched"][str(layer)][role]) for row in rows]
            aggregate = _aggregate(scores, "target_to_patched_kl_nats")
            margin_drop = target["mean_signed_margin"] - aggregate["mean_signed_margin"]
            removal = margin_drop / margin_gap if margin_gap > 1e-12 else None
            mass_retention = (
                aggregate["mean_label_mass"] / target["mean_label_mass"]
                if target["mean_label_mass"] > 0
                else None
            )
            passes = (
                removal is not None
                and removal >= CANDIDATE_REMOVAL_FRACTION
                and aggregate["format_rate"] >= MIN_FORMAT_RATE
                and mass_retention is not None
                and mass_retention >= MIN_LABEL_MASS_RETENTION
            )
            candidates[role][str(layer)] = {
                **aggregate,
                "mean_margin_drop_from_target": margin_drop,
                "mean_probability_drop_from_target": (
                    target["mean_conditional_correct_probability"]
                    - aggregate["mean_conditional_correct_probability"]
                ),
                "accuracy_drop_from_target": target["accuracy"] - aggregate["accuracy"],
                "label_mass_retention": mass_retention,
                "aggregate_margin_removal_fraction": removal,
                "passes_candidate_gate": passes,
            }

    selected: dict[str, list[int]] = {}
    for role in SYNTAX_ROLES:
        passing = [
            (layer, candidates[role][str(layer)]["aggregate_margin_removal_fraction"])
            for layer in LAYERS
            if candidates[role][str(layer)]["passes_candidate_gate"]
        ]
        passing.sort(key=lambda item: (-cast(float, item[1]), item[0]))
        selected[role] = [layer for layer, _effect in passing[:2]] if baseline_passes else []

    syntax_passes = any(selected.values())
    envelope_passes = any(
        candidates["all_positions"][str(layer)]["passes_candidate_gate"] for layer in LAYERS
    )
    only_envelope = baseline_passes and not syntax_passes and envelope_passes
    if not baseline_passes:
        stop_reason = "baseline_gate_failed"
    elif syntax_passes:
        stop_reason = None
    elif only_envelope:
        stop_reason = "only_all_positions_passed"
    else:
        stop_reason = "no_syntax_specific_role_passed"

    return {
        "schema_version": 1,
        "status": "DEV_ONLY_NOT_CONFIRMATORY",
        "protocol_sha256": protocol_sha,
        "raw_sha256": manifest["raw_sha256"],
        "config_sha256": manifest["config_sha256"],
        "analyzer_sha256": _sha256(Path(__file__)),
        "design_validation": design,
        "n_scored_forwards": manifest["n_scored_forwards"],
        "baseline": {
            "target": target,
            "test_only": test_only,
            "mean_margin_gap": margin_gap,
            "mean_conditional_correct_probability_gap": probability_gap,
            "accuracy_gap": accuracy_gap,
        },
        "layer_role": candidates,
        "selection": selected,
        "gates": {
            "baseline_probability_gap_at_least_0_15": probability_gap >= BASELINE_PROBABILITY_GAP,
            "baseline_accuracy_gap_at_least_0_25": accuracy_gap >= BASELINE_ACCURACY_GAP,
            "baseline_gate_passes": baseline_passes,
            "positive_baseline_margin_gap": margin_gap > 0,
            "syntax_specific_candidate_exists": syntax_passes,
            "all_positions_candidate_exists": envelope_passes,
            "only_all_positions_passes": only_envelope,
            "proceed_to_separately_frozen_head_screen": baseline_passes and syntax_passes,
            "stop": not (baseline_passes and syntax_passes),
            "stop_reason": stop_reason,
        },
        "analysis_scope": protocol["analysis_rules"],
        "format_validation": (
            "Every format_ok=True row is independently certified because its largest saved "
            "Q/K probability exceeds the total non-label probability mass; False is accepted "
            "conservatively."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path} or {temporary}")
    try:
        with temporary.open("x") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/attention_localization_dev_protocol_v2.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest or args.raw.with_suffix(".manifest.json")
    rows, manifest, protocol, protocol_sha = load_verified(args.raw, manifest_path, args.protocol)
    summary = analyse(rows, manifest, protocol, protocol_sha)
    _write_summary(args.out, summary)
    print(
        f"wrote {args.out}; proceed={summary['gates']['proceed_to_separately_frozen_head_screen']}"
    )


if __name__ == "__main__":
    main()
