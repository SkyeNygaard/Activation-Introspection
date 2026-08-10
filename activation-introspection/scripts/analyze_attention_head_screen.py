"""Fail-closed analysis for the frozen cross-DEV query-head screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect.codebook_icl import LABELS, VISIBLE_SAMPLES, Episode, exact_episodes, sha256_text

PROTOCOL_SHA256 = "759c0850d47d54ac4cc1fbce1ed3c53efd38f1b10c5a081a830936f69d25856d"
RUNNER_SHA256 = "93c8d4e26233ea3f081185d150ab9d6190c75aba17fd57f8a7e3d18f534841a0"
RUNNER_SOURCE = "scripts/run_attention_head_screen.py"
CONCEPTS = ("bread", "volcano", "violin")
CARRIER_INDICES = (1, 2)
CARRIERS = tuple(VISIBLE_SAMPLES[index] for index in CARRIER_INDICES)
PAIRS = (
    (21, "query_marker"),
    (23, "query_marker"),
    (26, "final_answer"),
    (31, "final_answer"),
)
PAIR_IDS = tuple(f"{role}@{layer}" for layer, role in PAIRS)
HEADS = tuple(range(16))
SENTINELS = (7, 8, 9)
CONTEXT_LAYERS = SENTINELS + tuple(layer for layer, _role in PAIRS)
CENTERING_CONCEPTS = (
    "ocean",
    "bread",
    "volcano",
    "violin",
    "spider",
    "hospital",
    "desert",
    "clock",
)
BASELINE_PROBABILITY_GAP = 0.15
BASELINE_ACCURACY_GAP = 0.25
PARENT_REMOVAL_FRACTION = 0.20
COMPONENT_REMOVAL_FRACTION = 0.10
MIN_FORMAT_RATE = 0.90
MIN_LABEL_MASS_RETENTION = 0.80
MIN_POSITIVE_STRATA = 5
_FLOAT_TOLERANCE = 2e-5
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}")
_COMMON_SCORE_KEYS = {
    "predicted_label",
    "correct",
    "signed_margin",
    "conditional_correct_probability",
    "label_logits",
    "full_logprobs",
    "label_mass",
    "format_ok",
    "full_logits_sha256",
    "full_argmax_token_id",
}
_CONFIG_KEYS = {
    "schema_version",
    "status",
    "estimand",
    "model_requested",
    "model_resolved",
    "model_revision",
    "device",
    "dtype",
    "concepts",
    "carriers",
    "carrier_indices",
    "cell_ids",
    "injection_layer",
    "strength",
    "pre_injection_sentinel_layers",
    "selected_layer_roles",
    "heads",
    "n_query_heads",
    "n_kv_heads",
    "head_width",
    "candidate_components",
    "scored_forwards_expected",
    "forward_mode",
    "offline_model_loading",
    "balanced_subset",
    "direction_sha256",
    "centering_direction_sha256",
    "centering_concepts",
    "max_offdiagonal_cosine",
    "prompt_set_sha256",
    "source_files_sha256",
    "source_sha256",
    "protocol",
    "protocol_sha256",
    "smoke",
    "selection_eligible",
    "publishable",
    "git_commit",
    "git_dirty",
    "python",
    "torch",
    "platform",
}
_ROW_KEYS = {
    "schema_version",
    "config_sha256",
    "unit_id",
    "concept",
    "carrier_id",
    "source_carrier_index",
    "cell_id",
    "episode_sha256",
    "prompt",
    "prompt_sha256",
    "token_ids",
    "token_ids_sha256",
    "label_token_ids",
    "state_token_positions",
    "query_marker_position",
    "demo_label_positions",
    "final_answer_position",
    "demo_signs",
    "query_sign",
    "label_mapping",
    "correct_label",
    "direction_sha256",
    "baseline",
    "instrumentation",
    "pairs",
}


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


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is not a valid integer")
    return value


def _dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} is not an object with string keys")
    return cast(dict[str, Any], value)


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys drifted")


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    digest = _sha256(path)
    if digest != PROTOCOL_SHA256:
        raise ValueError(f"protocol SHA-256 drifted: {digest}")
    protocol = _dict(json.loads(path.read_text()), "protocol")
    sources = _dict(protocol.get("source_files_sha256"), "protocol/source-files")
    if sources.get(RUNNER_SOURCE) != RUNNER_SHA256:
        raise ValueError("frozen protocol is not bound to the expected Stage 1b runner")
    root = Path(__file__).resolve().parents[1]
    for source, expected in sources.items():
        _hash(expected, f"protocol/source-files/{source}")
        source_path = (root / source).resolve()
        if not source_path.is_relative_to(root) or _sha256(source_path) != expected:
            raise ValueError(f"protocol-bound generation source drifted: {source}")
    return protocol, digest


def load_verified(
    raw_path: Path, manifest_path: Path, protocol_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    protocol, protocol_sha = load_protocol(protocol_path)
    manifest = _dict(json.loads(manifest_path.read_text()), "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "config",
            "config_sha256",
            "raw",
            "raw_sha256",
            "n_unit_rows",
            "n_scored_forwards",
        },
        "manifest",
    )
    config = _dict(manifest["config"], "manifest/config")
    config_sha = _hash(manifest["config_sha256"], "manifest/config-sha256")
    if _json_sha256(config) != config_sha:
        raise ValueError("manifest config SHA-256 does not match its config")
    if config.get("protocol") != protocol or config.get("protocol_sha256") != protocol_sha:
        raise ValueError("manifest config is not bound to the frozen protocol")
    if config.get("source_files_sha256") != protocol["source_files_sha256"]:
        raise ValueError("manifest source hashes differ from the protocol")
    if _json_sha256(config["source_files_sha256"]) != config.get("source_sha256"):
        raise ValueError("manifest aggregate source hash drifted")
    raw_sha = _sha256(raw_path)
    if raw_sha != _hash(manifest["raw_sha256"], "manifest/raw-sha256"):
        raise ValueError("raw artifact SHA-256 does not match its manifest")
    if manifest["raw"] != raw_path.name:
        raise ValueError("raw artifact filename does not match its manifest")
    lines = raw_path.read_text().splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("raw artifact contains an empty row")
    rows = [_dict(json.loads(line), f"raw/line-{index}") for index, line in enumerate(lines, 1)]
    if len(rows) != manifest["n_unit_rows"]:
        raise ValueError("raw row count does not match its manifest")
    if any(row.get("config_sha256") != config_sha for row in rows):
        raise ValueError("a raw row refers to a different config")
    return rows, manifest, protocol, protocol_sha


def _complementary_episodes(carrier: str) -> list[Episode]:
    episodes = exact_episodes(carrier)
    orders = sorted({episode.demo_signs for episode in episodes})
    selected = [
        episode
        for order_id, order in enumerate(orders)
        for episode in episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[1 - order_id % 2]
    ]
    if len(selected) != 12:
        raise ValueError("the expected complementary episode grid drifted")
    return selected


def _validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    _exact_keys(config, _CONFIG_KEYS, "config")
    design = _dict(protocol.get("design"), "protocol/design")
    expected = {
        "schema_version": 1,
        "status": "DEV_ONLY_CROSS_DEV_HEAD_SCREEN_NOT_CONFIRMATORY",
        "estimand": "cross-DEV single-query-head paired-interchange sensitivity",
        "model_requested": design["model"],
        "model_resolved": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": design["model_revision"],
        "device": "cpu",
        "dtype": "torch.float32",
        "concepts": list(CONCEPTS),
        "carriers": list(CARRIERS),
        "carrier_indices": list(CARRIER_INDICES),
        "cell_ids": [episode.cell_id for episode in _complementary_episodes(CARRIERS[0])],
        "injection_layer": 9,
        "strength": 1.0,
        "pre_injection_sentinel_layers": list(SENTINELS),
        "selected_layer_roles": [
            {"layer": layer, "role": role, "id": f"{role}@{layer}"} for layer, role in PAIRS
        ],
        "heads": list(HEADS),
        "n_query_heads": 16,
        "n_kv_heads": 2,
        "head_width": 128,
        "candidate_components": 64,
        "scored_forwards_expected": 5112,
        "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
        "offline_model_loading": True,
        "balanced_subset": "complementary mapping; six orders; both query-sign twins",
        "centering_concepts": list(CENTERING_CONCEPTS),
        "source_files_sha256": protocol["source_files_sha256"],
        "protocol": protocol,
        "protocol_sha256": PROTOCOL_SHA256,
        "smoke": False,
        "selection_eligible": True,
        "publishable": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"full-run config field {key!r} drifted")
    directions = _dict(config["direction_sha256"], "config/directions")
    if set(directions) != set(CONCEPTS):
        raise ValueError("configured direction grid drifted")
    for concept, digest in directions.items():
        _hash(digest, f"config/directions/{concept}")
    for key in ("centering_direction_sha256", "prompt_set_sha256", "source_sha256"):
        _hash(config[key], f"config/{key}")
    if config["source_sha256"] != _json_sha256(protocol["source_files_sha256"]):
        raise ValueError("configured aggregate source hash drifted")
    max_cosine = _number(config["max_offdiagonal_cosine"], "config/max-cosine", minimum=0)
    if max_cosine > 0.5:
        raise ValueError("configured DEV directions violate the cosine gate")
    if not isinstance(config["git_dirty"], bool):
        raise ValueError("config/git-dirty is not boolean")
    git_commit = config["git_commit"]
    if git_commit != "unknown" and (
        not isinstance(git_commit, str) or _GIT_RE.fullmatch(git_commit) is None
    ):
        raise ValueError("config/git-commit is invalid")
    for key in ("python", "torch", "platform"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"config/{key} is empty")


def _validate_score(
    score: dict[str, Any],
    *,
    correct_label: str,
    label_token_ids: dict[str, int],
    kl_key: str,
    location: str,
) -> None:
    _exact_keys(score, _COMMON_SCORE_KEYS | {kl_key}, location)
    if not isinstance(score["correct"], bool) or not isinstance(score["format_ok"], bool):
        raise ValueError(f"non-boolean score flag in {location}")
    logits = _dict(score["label_logits"], f"{location}/label-logits")
    logprobs = _dict(score["full_logprobs"], f"{location}/full-logprobs")
    if set(logits) != set(LABELS) or set(logprobs) != set(LABELS):
        raise ValueError(f"label score keys drifted in {location}")
    label_logits = {label: _number(logits[label], f"{location}/{label}-logit") for label in LABELS}
    label_logprobs = {
        label: _number(logprobs[label], f"{location}/{label}-logprob") for label in LABELS
    }
    if any(value > _FLOAT_TOLERANCE for value in label_logprobs.values()):
        raise ValueError(f"positive full-vocabulary log-probability in {location}")
    other_label = LABELS[1 - LABELS.index(correct_label)]
    margin = _number(score["signed_margin"], f"{location}/margin")
    if not math.isclose(
        margin,
        label_logits[correct_label] - label_logits[other_label],
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(f"signed margin is inconsistent in {location}")
    expected_probability = 1 / (1 + math.exp(-max(-700.0, min(700.0, margin))))
    probability = _probability(
        score["conditional_correct_probability"], f"{location}/correct-probability"
    )
    if not math.isclose(probability, expected_probability, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"conditional probability is inconsistent in {location}")
    masses = {label: math.exp(label_logprobs[label]) for label in LABELS}
    derived_mass = math.fsum(masses.values())
    if derived_mass <= 0:
        raise ValueError(f"label mass underflowed in {location}")
    label_mass = _probability(score["label_mass"], f"{location}/label-mass")
    if not math.isclose(label_mass, derived_mass, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"label mass is inconsistent in {location}")
    if not math.isclose(
        probability, masses[correct_label] / derived_mass, abs_tol=_FLOAT_TOLERANCE
    ):
        raise ValueError(f"label logits and full log-probabilities disagree in {location}")
    predicted = max(LABELS, key=label_logits.__getitem__)
    if score["predicted_label"] != predicted:
        raise ValueError(f"predicted label is inconsistent in {location}")
    if score["correct"] != (predicted == correct_label):
        raise ValueError(f"correctness flag is inconsistent in {location}")
    argmax = _integer(score["full_argmax_token_id"], f"{location}/full-argmax")
    expected_format = argmax in set(label_token_ids.values())
    if score["format_ok"] != expected_format:
        raise ValueError(f"format flag disagrees with the full-vocabulary argmax in {location}")
    if expected_format and argmax != label_token_ids[predicted]:
        raise ValueError(f"label and full-vocabulary argmax disagree in {location}")
    _hash(score["full_logits_sha256"], f"{location}/full-logits")
    if _number(score[kl_key], f"{location}/{kl_key}") < -1e-6:
        raise ValueError(f"negative KL divergence in {location}")


def validate_design(
    rows: list[dict[str, Any]], manifest: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, int]:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "config",
            "config_sha256",
            "raw",
            "raw_sha256",
            "n_unit_rows",
            "n_scored_forwards",
        },
        "manifest",
    )
    config = _dict(manifest["config"], "manifest/config")
    _validate_config(config, protocol)
    config_sha = _hash(manifest["config_sha256"], "manifest/config-sha256")
    if _json_sha256(config) != config_sha:
        raise ValueError("manifest config SHA-256 does not match its config")
    _hash(manifest["raw_sha256"], "manifest/raw-sha256")
    if (
        manifest["schema_version"] != 1
        or manifest["n_unit_rows"] != 72
        or manifest["n_scored_forwards"] != 5112
    ):
        raise ValueError("manifest schema or frozen compute budget drifted")

    expected: dict[tuple[str, int, str], tuple[int, Episode]] = {}
    for concept in CONCEPTS:
        for carrier_id, (carrier_index, carrier) in enumerate(
            zip(CARRIER_INDICES, CARRIERS, strict=True)
        ):
            for episode in _complementary_episodes(carrier):
                expected[(concept, carrier_index, episode.cell_id)] = (carrier_id, episode)
    if len(rows) != len(expected):
        raise ValueError("the exact 72-unit cross-DEV grid is incomplete")

    seen: set[tuple[str, int, str]] = set()
    prompt_hashes: list[str] = []
    twin_fingerprints: dict[tuple[str, int, str], list[tuple[object, ...]]] = {}
    label_id_sets: set[tuple[int, int]] = set()
    stratum_counts = {(concept, carrier): 0 for concept in CONCEPTS for carrier in CARRIER_INDICES}
    patch_count = 0
    for row in rows:
        _exact_keys(row, _ROW_KEYS, "row")
        if row["schema_version"] != 1 or row["config_sha256"] != manifest["config_sha256"]:
            raise ValueError("row schema or config binding drifted")
        concept = row["concept"]
        carrier_index = row["source_carrier_index"]
        cell_id = row["cell_id"]
        if not isinstance(concept, str) or not isinstance(cell_id, str):
            raise ValueError("row unit identity is not textual")
        _integer(carrier_index, "row/source-carrier-index")
        key = (concept, carrier_index, cell_id)
        if key not in expected or key in seen:
            raise ValueError("row unit grid contains an unexpected or duplicate unit")
        seen.add(key)
        carrier_id, episode = expected[key]
        location = f"unit/{concept}/carrier-{carrier_index}/{cell_id}"
        if (
            row["unit_id"] != f"{concept}/carrier-{carrier_index}/{cell_id}"
            or row["carrier_id"] != carrier_id
            or row["episode_sha256"] != episode.digest()
            or tuple(row["demo_signs"]) != episode.demo_signs
            or row["query_sign"] != episode.query_sign
            or row["label_mapping"] != {"+1": episode.positive_label, "-1": episode.negative_label}
            or row["correct_label"] != episode.correct_label
            or row["direction_sha256"] != config["direction_sha256"][concept]
        ):
            raise ValueError(f"unit fields drifted in {location}")
        stratum_counts[(concept, carrier_index)] += 1

        prompt = row["prompt"]
        if (
            not isinstance(prompt, str)
            or sha256_text(prompt) != row["prompt_sha256"]
            or prompt.count(episode.render_user()) != 1
        ):
            raise ValueError(f"prompt binding drifted in {location}")
        prompt_hashes.append(row["prompt_sha256"])
        token_ids = row["token_ids"]
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in token_ids
            )
            or _json_sha256(token_ids) != row["token_ids_sha256"]
        ):
            raise ValueError(f"token binding drifted in {location}")
        label_tokens = _dict(row["label_token_ids"], f"{location}/label-token-ids")
        if set(label_tokens) != set(LABELS):
            raise ValueError(f"label-token grid drifted in {location}")
        label_token_ids = {
            label: _integer(label_tokens[label], f"{location}/{label}-token") for label in LABELS
        }
        if len(set(label_token_ids.values())) != 2:
            raise ValueError(f"label token IDs collide in {location}")
        label_id_sets.add((label_token_ids[LABELS[0]], label_token_ids[LABELS[1]]))

        state_positions = row["state_token_positions"]
        demo_positions = row["demo_label_positions"]
        if (
            not isinstance(state_positions, list)
            or len(state_positions) != 5
            or any(
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 0 <= position < len(token_ids)
                for position in state_positions
            )
            or state_positions != sorted(set(state_positions))
            or not isinstance(demo_positions, list)
            or len(demo_positions) != 4
            or any(
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 0 <= position < len(token_ids)
                for position in demo_positions
            )
            or demo_positions != sorted(set(demo_positions))
        ):
            raise ValueError(f"receiver positions drifted in {location}")
        answer = _integer(row["final_answer_position"], f"{location}/answer-position")
        if (
            row["query_marker_position"] != state_positions[-1]
            or not all(
                marker < label
                for marker, label in zip(state_positions[:4], demo_positions, strict=True)
            )
            or not max(demo_positions) < state_positions[-1] < answer == len(token_ids) - 1
        ):
            raise ValueError(f"frozen causal ordering drifted in {location}")
        expected_demo_labels = [episode.label_for(sign) for sign in episode.demo_signs]
        if any(
            token_ids[position] != label_token_ids[label]
            for position, label in zip(demo_positions, expected_demo_labels, strict=True)
        ):
            raise ValueError(f"demonstration-label token binding drifted in {location}")
        twin_key = (concept, carrier_index, cell_id.rsplit("q", 1)[0])
        twin_fingerprints.setdefault(twin_key, []).append(
            (
                row["prompt_sha256"],
                row["token_ids_sha256"],
                tuple(state_positions),
                tuple(demo_positions),
                tuple(label_token_ids[label] for label in LABELS),
            )
        )

        baseline = _dict(row["baseline"], f"{location}/baseline")
        _exact_keys(baseline, {"target", "test_only"}, f"{location}/baseline")
        target = _dict(baseline["target"], f"{location}/target")
        test_only = _dict(baseline["test_only"], f"{location}/test-only")
        _validate_score(
            target,
            correct_label=episode.correct_label,
            label_token_ids=label_token_ids,
            kl_key="target_to_target_kl_nats",
            location=f"{location}/target",
        )
        _validate_score(
            test_only,
            correct_label=episode.correct_label,
            label_token_ids=label_token_ids,
            kl_key="target_to_test_only_kl_nats",
            location=f"{location}/test-only",
        )
        if target["target_to_target_kl_nats"] != 0.0:
            raise ValueError(f"target self-KL is not exactly zero in {location}")

        instrumentation = _dict(row["instrumentation"], f"{location}/instrumentation")
        _exact_keys(
            instrumentation,
            {
                "pre_injection_sentinels_equal",
                "self_patch_score",
                "self_patch_exact",
                "self_patch_max_abs_logit_error",
                "attention_context_sha256",
            },
            f"{location}/instrumentation",
        )
        self_score = _dict(instrumentation["self_patch_score"], f"{location}/self-patch")
        _validate_score(
            self_score,
            correct_label=episode.correct_label,
            label_token_ids=label_token_ids,
            kl_key="target_to_self_patch_kl_nats",
            location=f"{location}/self-patch",
        )
        target_common = {key: target[key] for key in _COMMON_SCORE_KEYS}
        self_common = {key: self_score[key] for key in _COMMON_SCORE_KEYS}
        if (
            instrumentation["pre_injection_sentinels_equal"] is not True
            or instrumentation["self_patch_exact"] is not True
            or instrumentation["self_patch_max_abs_logit_error"] != 0.0
            or self_score["target_to_self_patch_kl_nats"] != 0.0
            or self_common != target_common
        ):
            raise ValueError(f"exact self-donor or sentinel gate failed in {location}")
        contexts = _dict(
            instrumentation["attention_context_sha256"], f"{location}/attention-contexts"
        )
        _exact_keys(contexts, {"target", "test_only"}, f"{location}/attention-contexts")
        expected_context_keys = {str(layer) for layer in CONTEXT_LAYERS}
        for condition in ("target", "test_only"):
            hashes = _dict(contexts[condition], f"{location}/{condition}-contexts")
            _exact_keys(hashes, expected_context_keys, f"{location}/{condition}-contexts")
            for context_layer, digest in hashes.items():
                _hash(digest, f"{location}/{condition}-context/{context_layer}")
        if any(
            contexts["target"][str(layer)] != contexts["test_only"][str(layer)]
            for layer in SENTINELS
        ):
            raise ValueError(f"pre-injection context hashes differ in {location}")

        pairs = _dict(row["pairs"], f"{location}/pairs")
        _exact_keys(pairs, set(PAIR_IDS), f"{location}/pairs")
        for layer, role in PAIRS:
            pair_id = f"{role}@{layer}"
            pair = _dict(pairs[pair_id], f"{location}/{pair_id}")
            _exact_keys(pair, {"layer", "role", "all_heads", "heads"}, f"{location}/{pair_id}")
            if pair["layer"] != layer or pair["role"] != role:
                raise ValueError(f"pair identity drifted in {location}/{pair_id}")
            _validate_score(
                _dict(pair["all_heads"], f"{location}/{pair_id}/all-heads"),
                correct_label=episode.correct_label,
                label_token_ids=label_token_ids,
                kl_key="target_to_patched_kl_nats",
                location=f"{location}/{pair_id}/all-heads",
            )
            heads = _dict(pair["heads"], f"{location}/{pair_id}/heads")
            _exact_keys(heads, {str(head) for head in HEADS}, f"{location}/{pair_id}/heads")
            for head in HEADS:
                _validate_score(
                    _dict(heads[str(head)], f"{location}/{pair_id}/head-{head}"),
                    correct_label=episode.correct_label,
                    label_token_ids=label_token_ids,
                    kl_key="target_to_patched_kl_nats",
                    location=f"{location}/{pair_id}/head-{head}",
                )
                patch_count += 1
            patch_count += 1

    if seen != set(expected) or any(count != 12 for count in stratum_counts.values()):
        raise ValueError("the six-stratum unit grid is incomplete")
    if len(label_id_sets) != 1:
        raise ValueError("label token IDs drift across units")
    if len(twin_fingerprints) != 36 or any(
        len(values) != 2 or len(set(values)) != 1 for values in twin_fingerprints.values()
    ):
        raise ValueError("byte-identical query twins are incomplete")
    if _json_sha256(sorted(prompt_hashes)) != config["prompt_set_sha256"]:
        raise ValueError("prompt-set SHA-256 drifted")
    if patch_count != 72 * 4 * 17:
        raise ValueError("72 x 4 x 17 patch grid is incomplete")
    return {
        "units": 72,
        "strata": 6,
        "cells_per_stratum": 12,
        "parents": 4,
        "components": 64,
        "patched_forwards": patch_count,
        "scored_forwards": 5112,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _strata(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        f"{concept}/carrier-{carrier_index}": []
        for concept in CONCEPTS
        for carrier_index in CARRIER_INDICES
    }
    for row in rows:
        grouped[f"{row['concept']}/carrier-{row['source_carrier_index']}"].append(row)
    if any(len(group) != 12 for group in grouped.values()):
        raise ValueError("cannot aggregate an incomplete stratum")
    return grouped


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


ScoreGetter = Callable[[dict[str, Any]], dict[str, Any]]


def _summarize_scores(
    grouped: dict[str, list[dict[str, Any]]], getter: ScoreGetter, kl_key: str
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    by_stratum = {
        stratum: _aggregate([getter(row) for row in stratum_rows], kl_key)
        for stratum, stratum_rows in grouped.items()
    }
    aggregate = {
        metric: _mean([stats[metric] for stats in by_stratum.values()])
        for metric in next(iter(by_stratum.values()))
    }
    return aggregate, by_stratum


def _paired_margin_differences(
    grouped: dict[str, list[dict[str, Any]]], left: ScoreGetter, right: ScoreGetter
) -> dict[str, float]:
    return {
        stratum: _mean(
            [
                float(left(row)["signed_margin"]) - float(right(row)["signed_margin"])
                for row in group
            ]
        )
        for stratum, group in grouped.items()
    }


def _effect(
    grouped: dict[str, list[dict[str, Any]]],
    target_getter: ScoreGetter,
    patch_getter: ScoreGetter,
    *,
    target: dict[str, float],
    baseline_margin_gap: float,
    removal_threshold: float,
) -> dict[str, Any]:
    patched, by_stratum = _summarize_scores(grouped, patch_getter, "target_to_patched_kl_nats")
    stratum_drops = _paired_margin_differences(grouped, target_getter, patch_getter)
    margin_drop = _mean(list(stratum_drops.values()))
    removal = margin_drop / baseline_margin_gap if baseline_margin_gap > 0 else None
    mass_retention = patched["mean_label_mass"] / target["mean_label_mass"]
    positive_strata = [stratum for stratum, drop in stratum_drops.items() if drop > 0]
    gates = {
        "aggregate_margin_removal_at_least_threshold": (
            removal is not None and removal >= removal_threshold
        ),
        "format_rate_at_least_0_90": patched["format_rate"] >= MIN_FORMAT_RATE,
        "label_mass_retention_at_least_0_80": mass_retention >= MIN_LABEL_MASS_RETENTION,
        "positive_margin_drop_in_at_least_5_of_6_strata": (
            len(positive_strata) >= MIN_POSITIVE_STRATA
        ),
    }
    return {
        "aggregate": patched,
        "by_stratum": by_stratum,
        "mean_margin_drop_from_target": margin_drop,
        "aggregate_margin_removal_fraction": removal,
        "label_mass_retention": mass_retention,
        "margin_drop_by_stratum": stratum_drops,
        "positive_margin_drop_strata": positive_strata,
        "n_positive_margin_drop_strata": len(positive_strata),
        "removal_threshold": removal_threshold,
        "gates": gates,
        "passes": all(gates.values()),
    }


def analyse(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    if protocol_sha != PROTOCOL_SHA256:
        raise ValueError("analysis was not given the frozen Stage 1b protocol")
    design_validation = validate_design(rows, manifest, protocol)
    grouped = _strata(rows)

    def target_getter(row: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], row["baseline"]["target"])

    def test_getter(row: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], row["baseline"]["test_only"])

    target, target_strata = _summarize_scores(grouped, target_getter, "target_to_target_kl_nats")
    test_only, test_strata = _summarize_scores(grouped, test_getter, "target_to_test_only_kl_nats")
    margin_gaps = _paired_margin_differences(grouped, target_getter, test_getter)
    margin_gap = _mean(list(margin_gaps.values()))
    probability_gap = (
        target["mean_conditional_correct_probability"]
        - test_only["mean_conditional_correct_probability"]
    )
    accuracy_gap = target["accuracy"] - test_only["accuracy"]
    positive_baseline_strata = [stratum for stratum, gap in margin_gaps.items() if gap > 0]
    baseline_gates = {
        "conditional_correct_probability_gap_at_least_0_15": (
            probability_gap >= BASELINE_PROBABILITY_GAP
        ),
        "accuracy_gap_at_least_0_25": accuracy_gap >= BASELINE_ACCURACY_GAP,
        "positive_aggregate_margin_denominator": margin_gap > 0,
        "positive_margin_gap_in_at_least_5_of_6_strata": (
            len(positive_baseline_strata) >= MIN_POSITIVE_STRATA
        ),
    }
    baseline_passes = all(baseline_gates.values())

    parents: dict[str, dict[str, Any]] = {}
    components: dict[str, dict[str, Any]] = {}
    for layer, role in PAIRS:
        pair_id = f"{role}@{layer}"

        def parent_getter(row: dict[str, Any], pair_id: str = pair_id) -> dict[str, Any]:
            return cast(dict[str, Any], row["pairs"][pair_id]["all_heads"])

        parents[pair_id] = {
            "layer": layer,
            "receiver_role": role,
            **_effect(
                grouped,
                target_getter,
                parent_getter,
                target=target,
                baseline_margin_gap=margin_gap,
                removal_threshold=PARENT_REMOVAL_FRACTION,
            ),
        }
        for head in HEADS:
            component_id = f"{pair_id}/head-{head}"

            def component_getter(
                row: dict[str, Any], pair_id: str = pair_id, head: int = head
            ) -> dict[str, Any]:
                return cast(dict[str, Any], row["pairs"][pair_id]["heads"][str(head)])

            components[component_id] = {
                "layer": layer,
                "receiver_role": role,
                "query_head": head,
                **_effect(
                    grouped,
                    target_getter,
                    component_getter,
                    target=target,
                    baseline_margin_gap=margin_gap,
                    removal_threshold=COMPONENT_REMOVAL_FRACTION,
                ),
            }

    qualifying = [component_id for component_id, result in components.items() if result["passes"]]
    represented_families = sorted({components[item]["receiver_role"] for item in qualifying})
    parents_pass = all(result["passes"] for result in parents.values())
    count_passes = 2 <= len(qualifying) <= 4
    families_pass = set(represented_families) == {"query_marker", "final_answer"}
    proceed = baseline_passes and parents_pass and count_passes and families_pass
    stop_reasons: list[str] = []
    if not baseline_passes:
        stop_reasons.append("baseline_gate_failed")
    stop_reasons.extend(
        f"all_head_parent_failed:{pair_id}"
        for pair_id, result in parents.items()
        if not result["passes"]
    )
    if not count_passes:
        stop_reasons.append("qualifying_component_count_outside_2_to_4")
    for family in ("query_marker", "final_answer"):
        if family not in represented_families:
            stop_reasons.append(f"missing_receiver_family:{family}")

    return {
        "schema_version": 1,
        "status": "DEV_ONLY_SELECTION_ONLY_NOT_CONFIRMATORY",
        "protocol_sha256": protocol_sha,
        "runner_sha256": RUNNER_SHA256,
        "raw_sha256": manifest["raw_sha256"],
        "config_sha256": manifest["config_sha256"],
        "analyzer_sha256": _sha256(Path(__file__)),
        "design_validation": design_validation,
        "n_scored_forwards": manifest["n_scored_forwards"],
        "baseline": {
            "target": target,
            "test_only": test_only,
            "target_by_stratum": target_strata,
            "test_only_by_stratum": test_strata,
            "mean_margin_gap": margin_gap,
            "mean_conditional_correct_probability_gap": probability_gap,
            "accuracy_gap": accuracy_gap,
            "margin_gap_by_stratum": margin_gaps,
            "positive_margin_gap_strata": positive_baseline_strata,
            "gates": baseline_gates,
            "passes": baseline_passes,
        },
        "all_head_replications": parents,
        "components": components,
        "selection": {
            "qualifying_components": qualifying,
            "n_qualifying_components": len(qualifying),
            "represented_receiver_families": represented_families,
            "all_passers_retained_without_top_k_truncation": True,
        },
        "gates": {
            "baseline_passes": baseline_passes,
            "all_four_all_head_replications_pass": parents_pass,
            "qualifying_component_count_between_2_and_4": count_passes,
            "both_receiver_families_represented": families_pass,
            "proceed_to_separately_frozen_stage1c": proceed,
            "stop": not proceed,
            "stop_reasons": stop_reasons,
        },
        "analysis_scope": protocol["analysis_rules"],
        "stage1c_handoff": (
            "A go authorizes only a separately frozen envelope, reciprocal-rescue, "
            "joint/complement, and zero/random-control study. This screen is selection only."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with temporary.open("x") as handle:
            created = True
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/attention_head_screen_dev_protocol_v3.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest or args.raw.with_suffix(".manifest.json")
    rows, manifest, protocol, protocol_sha = load_verified(args.raw, manifest_path, args.protocol)
    summary = analyse(rows, manifest, protocol, protocol_sha)
    _write_summary(args.out, summary)
    print(
        f"wrote {args.out}; "
        f"proceed={summary['gates']['proceed_to_separately_frozen_stage1c']}; "
        f"components={summary['selection']['n_qualifying_components']}"
    )


if __name__ == "__main__":
    main()
