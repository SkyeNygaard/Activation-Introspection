from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from introspect.codebook_icl import LABELS, exact_episodes, sha256_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_attention_head_screen as analysis

LABEL_TOKEN_IDS = {"Q": ord("Q"), "K": ord("K")}


def _digest(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _score(
    correct_label: str,
    margin: float,
    tag: str,
    kl_key: str,
    *,
    label_mass: float = 0.9,
    format_ok: bool = True,
) -> dict[str, Any]:
    correct_probability = 1 / (1 + math.exp(-max(-700.0, min(700.0, margin))))
    other_label = LABELS[1 - LABELS.index(correct_label)]
    probabilities = {correct_label: correct_probability, other_label: 1 - correct_probability}
    logits = {correct_label: margin / 2, other_label: -margin / 2}
    predicted = max(LABELS, key=logits.__getitem__)
    return {
        "predicted_label": predicted,
        "correct": predicted == correct_label,
        "signed_margin": margin,
        "conditional_correct_probability": correct_probability,
        "label_logits": logits,
        "full_logprobs": {label: math.log(label_mass * probabilities[label]) for label in LABELS},
        "label_mass": label_mass,
        "format_ok": format_ok,
        "full_logits_sha256": _digest(tag),
        "full_argmax_token_id": LABEL_TOKEN_IDS[predicted] if format_ok else 999,
        kl_key: 0.0 if "target_to_target" in kl_key else 0.1,
    }


def _complementary_episodes(carrier: str) -> list[Any]:
    episodes = exact_episodes(carrier)
    orders = sorted({episode.demo_signs for episode in episodes})
    return [
        episode
        for order_id, order in enumerate(orders)
        for episode in episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[1 - order_id % 2]
    ]


def _artifact(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    protocol_path = root / "results/attention_head_screen_dev_protocol_v3.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    directions = {concept: _digest(f"direction/{concept}") for concept in analysis.CONCEPTS}
    for concept in analysis.CONCEPTS:
        for carrier_id, (carrier_index, carrier) in enumerate(
            zip(analysis.CARRIER_INDICES, analysis.CARRIERS, strict=True)
        ):
            for episode in _complementary_episodes(carrier):
                prompt = f"<chat>{episode.render_user()}</chat>Label:"
                token_ids = [ord(character) for character in prompt]
                state_positions = [match.start() for match in re.finditer("§", prompt)]
                demo_positions = [match.start(1) for match in re.finditer(r"Label: ([QK])", prompt)]
                unit = f"{concept}/carrier-{carrier_index}/{episode.cell_id}"
                target = _score(
                    episode.correct_label,
                    2.0,
                    f"{unit}/target",
                    "target_to_target_kl_nats",
                )
                test_only = _score(
                    episode.correct_label,
                    0.0,
                    f"{unit}/test-only",
                    "target_to_test_only_kl_nats",
                )
                pairs: dict[str, Any] = {}
                for layer, role in analysis.PAIRS:
                    pair_id = f"{role}@{layer}"
                    all_heads = _score(
                        episode.correct_label,
                        1.4,
                        f"{unit}/{pair_id}/all",
                        "target_to_patched_kl_nats",
                    )
                    heads = {}
                    for head in analysis.HEADS:
                        selected = (pair_id, head) in {
                            ("query_marker@21", 0),
                            ("final_answer@26", 0),
                        }
                        heads[str(head)] = _score(
                            episode.correct_label,
                            1.6 if selected else 2.0,
                            f"{unit}/{pair_id}/head-{head}",
                            "target_to_patched_kl_nats",
                        )
                    pairs[pair_id] = {
                        "layer": layer,
                        "role": role,
                        "all_heads": all_heads,
                        "heads": heads,
                    }
                target_contexts = {
                    str(layer): _digest(f"{unit}/target-context/{layer}")
                    for layer in analysis.CONTEXT_LAYERS
                }
                test_contexts = {
                    str(layer): (
                        target_contexts[str(layer)]
                        if layer in analysis.SENTINELS
                        else _digest(f"{unit}/test-context/{layer}")
                    )
                    for layer in analysis.CONTEXT_LAYERS
                }
                self_score = copy.deepcopy(target)
                self_score["target_to_self_patch_kl_nats"] = 0.0
                self_score.pop("target_to_target_kl_nats")
                row: dict[str, Any] = {
                    "schema_version": 1,
                    "config_sha256": "",
                    "unit_id": unit,
                    "concept": concept,
                    "carrier_id": carrier_id,
                    "source_carrier_index": carrier_index,
                    "cell_id": episode.cell_id,
                    "episode_sha256": episode.digest(),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "token_ids": token_ids,
                    "token_ids_sha256": analysis._json_sha256(token_ids),
                    "label_token_ids": LABEL_TOKEN_IDS,
                    "state_token_positions": state_positions,
                    "query_marker_position": state_positions[-1],
                    "demo_label_positions": demo_positions,
                    "final_answer_position": len(token_ids) - 1,
                    "demo_signs": list(episode.demo_signs),
                    "query_sign": episode.query_sign,
                    "label_mapping": {
                        "+1": episode.positive_label,
                        "-1": episode.negative_label,
                    },
                    "correct_label": episode.correct_label,
                    "direction_sha256": directions[concept],
                    "baseline": {"target": target, "test_only": test_only},
                    "instrumentation": {
                        "pre_injection_sentinels_equal": True,
                        "self_patch_score": self_score,
                        "self_patch_exact": True,
                        "self_patch_max_abs_logit_error": 0.0,
                        "attention_context_sha256": {
                            "target": target_contexts,
                            "test_only": test_contexts,
                        },
                    },
                    "pairs": pairs,
                }
                rows.append(row)
                prompt_hashes.append(row["prompt_sha256"])

    config: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEV_ONLY_CROSS_DEV_HEAD_SCREEN_NOT_CONFIRMATORY",
        "estimand": "cross-DEV single-query-head paired-interchange sensitivity",
        "model_requested": protocol["design"]["model"],
        "model_resolved": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": protocol["design"]["model_revision"],
        "device": "cpu",
        "dtype": "torch.float32",
        "concepts": list(analysis.CONCEPTS),
        "carriers": list(analysis.CARRIERS),
        "carrier_indices": list(analysis.CARRIER_INDICES),
        "cell_ids": [episode.cell_id for episode in _complementary_episodes(analysis.CARRIERS[0])],
        "injection_layer": 9,
        "strength": 1.0,
        "pre_injection_sentinel_layers": list(analysis.SENTINELS),
        "selected_layer_roles": [
            {"layer": layer, "role": role, "id": f"{role}@{layer}"}
            for layer, role in analysis.PAIRS
        ],
        "heads": list(analysis.HEADS),
        "n_query_heads": 16,
        "n_kv_heads": 2,
        "head_width": 128,
        "candidate_components": 64,
        "scored_forwards_expected": 5112,
        "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
        "offline_model_loading": True,
        "balanced_subset": "complementary mapping; six orders; both query-sign twins",
        "direction_sha256": directions,
        "centering_direction_sha256": _digest("centering-direction"),
        "centering_concepts": list(analysis.CENTERING_CONCEPTS),
        "max_offdiagonal_cosine": 0.2,
        "prompt_set_sha256": analysis._json_sha256(sorted(prompt_hashes)),
        "source_files_sha256": protocol["source_files_sha256"],
        "source_sha256": analysis._json_sha256(protocol["source_files_sha256"]),
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "smoke": False,
        "selection_eligible": True,
        "publishable": False,
        "git_commit": "a" * 40,
        "git_dirty": True,
        "python": "3.12.synthetic",
        "torch": "2.synthetic",
        "platform": "synthetic",
    }
    config_sha = analysis._json_sha256(config)
    for row in rows:
        row["config_sha256"] = config_sha
    manifest = {
        "schema_version": 1,
        "config": config,
        "config_sha256": config_sha,
        "raw": "synthetic.jsonl",
        "raw_sha256": _digest("synthetic-raw"),
        "n_unit_rows": 72,
        "n_scored_forwards": 5112,
    }
    return rows, manifest, protocol, protocol_sha


def _replace_component(
    row: dict[str, Any],
    pair_id: str,
    head: int,
    margin: float,
    *,
    label_mass: float = 0.9,
    format_ok: bool = True,
) -> None:
    row["pairs"][pair_id]["heads"][str(head)] = _score(
        row["correct_label"],
        margin,
        f"replacement/{row['unit_id']}/{pair_id}/{head}/{margin}/{label_mass}/{format_ok}",
        "target_to_patched_kl_nats",
        label_mass=label_mass,
        format_ok=format_ok,
    )


def _replace_target(row: dict[str, Any], margin: float) -> None:
    target = _score(
        row["correct_label"],
        margin,
        f"replacement/{row['unit_id']}/target/{margin}",
        "target_to_target_kl_nats",
    )
    self_score = copy.deepcopy(target)
    self_score["target_to_self_patch_kl_nats"] = 0.0
    self_score.pop("target_to_target_kl_nats")
    row["baseline"]["target"] = target
    row["instrumentation"]["self_patch_score"] = self_score


def test_valid_artifact_binds_and_selects_every_passer(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest, protocol, protocol_sha = _artifact(root)
    raw_path = tmp_path / "synthetic.jsonl"
    manifest_path = tmp_path / "synthetic.manifest.json"
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    raw_path.write_text(raw)
    manifest["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    loaded, loaded_manifest, loaded_protocol, loaded_sha = analysis.load_verified(
        raw_path,
        manifest_path,
        root / "results/attention_head_screen_dev_protocol_v3.json",
    )
    summary = analysis.analyse(loaded, loaded_manifest, loaded_protocol, loaded_sha)

    assert loaded_protocol == protocol and loaded_sha == protocol_sha
    assert summary["design_validation"] == {
        "units": 72,
        "strata": 6,
        "cells_per_stratum": 12,
        "parents": 4,
        "components": 64,
        "patched_forwards": 4896,
        "scored_forwards": 5112,
    }
    assert all(result["passes"] for result in summary["all_head_replications"].values())
    assert summary["selection"]["qualifying_components"] == [
        "query_marker@21/head-0",
        "final_answer@26/head-0",
    ]
    assert summary["components"]["query_marker@21/head-0"][
        "aggregate_margin_removal_fraction"
    ] == pytest.approx(0.2)
    assert summary["gates"]["proceed_to_separately_frozen_stage1c"] is True
    assert "p_value" not in json.dumps(summary)
    assert summary["status"] == "DEV_ONLY_SELECTION_ONLY_NOT_CONFIRMATORY"

    summary_path = tmp_path / "summary.json"
    analysis._write_summary(summary_path, summary)
    assert json.loads(summary_path.read_text()) == summary
    with pytest.raises(FileExistsError):
        analysis._write_summary(summary_path, summary)

    raw_path.write_text(raw + " ")
    with pytest.raises(ValueError, match="raw artifact SHA-256"):
        analysis.load_verified(
            raw_path,
            manifest_path,
            root / "results/attention_head_screen_dev_protocol_v3.json",
        )


def test_validation_recomputes_scores_and_rejects_instrumentation_tampering() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest, protocol, _protocol_sha = _artifact(root)

    broken_score = copy.deepcopy(rows)
    broken_score[0]["pairs"]["query_marker@21"]["heads"]["0"]["signed_margin"] += 0.1
    with pytest.raises(ValueError, match="signed margin is inconsistent"):
        analysis.validate_design(broken_score, manifest, protocol)

    broken_format = copy.deepcopy(rows)
    broken_format[0]["pairs"]["query_marker@21"]["heads"]["0"]["full_argmax_token_id"] = 999
    with pytest.raises(ValueError, match="format flag disagrees"):
        analysis.validate_design(broken_format, manifest, protocol)

    broken_sham = copy.deepcopy(rows)
    broken_sham[0]["instrumentation"]["self_patch_score"]["full_logits_sha256"] = _digest(
        "wrong-sham"
    )
    with pytest.raises(ValueError, match="exact self-donor"):
        analysis.validate_design(broken_sham, manifest, protocol)

    broken_sentinel = copy.deepcopy(rows)
    broken_sentinel[0]["instrumentation"]["attention_context_sha256"]["test_only"][
        str(analysis.SENTINELS[0])
    ] = _digest("wrong-sentinel")
    with pytest.raises(ValueError, match="pre-injection context hashes differ"):
        analysis.validate_design(broken_sentinel, manifest, protocol)

    broken_grid = copy.deepcopy(rows)
    del broken_grid[0]["pairs"]["query_marker@21"]["heads"]["15"]
    with pytest.raises(ValueError, match="heads keys drifted"):
        analysis.validate_design(broken_grid, manifest, protocol)


def test_baseline_and_replication_gates_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest, protocol, protocol_sha = _artifact(root)

    negative_denominator = copy.deepcopy(rows)
    for index, row in enumerate(negative_denominator):
        if index % 12 < 3:
            _replace_target(row, -100.0)
    stopped = analysis.analyse(negative_denominator, manifest, protocol, protocol_sha)
    assert stopped["baseline"]["mean_conditional_correct_probability_gap"] >= 0.15
    assert stopped["baseline"]["accuracy_gap"] >= 0.25
    assert stopped["baseline"]["mean_margin_gap"] < 0
    assert stopped["baseline"]["passes"] is False
    assert all(
        result["aggregate_margin_removal_fraction"] is None
        for result in stopped["components"].values()
    )
    assert stopped["gates"]["proceed_to_separately_frozen_stage1c"] is False

    parent_below_threshold = copy.deepcopy(rows)
    for row in parent_below_threshold:
        row["pairs"]["query_marker@21"]["all_heads"] = _score(
            row["correct_label"],
            2.0 - 2.0 * 0.19999,
            f"below-parent/{row['unit_id']}",
            "target_to_patched_kl_nats",
        )
    parent_stop = analysis.analyse(parent_below_threshold, manifest, protocol, protocol_sha)
    parent = parent_stop["all_head_replications"]["query_marker@21"]
    assert parent["aggregate_margin_removal_fraction"] == pytest.approx(0.19999)
    assert parent["passes"] is False
    assert "all_head_parent_failed:query_marker@21" in parent_stop["gates"]["stop_reasons"]


def test_component_thresholds_and_stratum_stability_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest, protocol, protocol_sha = _artifact(root)

    five_positive = copy.deepcopy(rows)
    last_stratum = (analysis.CONCEPTS[-1], analysis.CARRIER_INDICES[-1])
    for row in five_positive:
        if (row["concept"], row["source_carrier_index"]) == last_stratum:
            _replace_component(row, "query_marker@21", 0, 2.01)
    five = analysis.analyse(five_positive, manifest, protocol, protocol_sha)
    component = five["components"]["query_marker@21/head-0"]
    assert component["n_positive_margin_drop_strata"] == 5
    assert component["aggregate_margin_removal_fraction"] >= 0.10
    assert component["passes"] is True

    four_positive = copy.deepcopy(rows)
    nonpositive = {
        (analysis.CONCEPTS[-1], analysis.CARRIER_INDICES[-1]),
        (analysis.CONCEPTS[-1], analysis.CARRIER_INDICES[0]),
    }
    for row in four_positive:
        if (row["concept"], row["source_carrier_index"]) in nonpositive:
            _replace_component(row, "query_marker@21", 0, 2.01)
    four = analysis.analyse(four_positive, manifest, protocol, protocol_sha)
    component = four["components"]["query_marker@21/head-0"]
    assert component["n_positive_margin_drop_strata"] == 4
    assert component["aggregate_margin_removal_fraction"] >= 0.10
    assert component["passes"] is False

    exact_mass = copy.deepcopy(rows)
    for row in exact_mass:
        _replace_component(row, "query_marker@21", 0, 1.6, label_mass=0.72)
    exact = analysis.analyse(exact_mass, manifest, protocol, protocol_sha)
    assert exact["components"]["query_marker@21/head-0"]["label_mass_retention"] == pytest.approx(
        0.8
    )
    assert exact["components"]["query_marker@21/head-0"]["passes"] is True

    below_mass = copy.deepcopy(rows)
    for row in below_mass:
        _replace_component(row, "query_marker@21", 0, 1.6, label_mass=0.719)
    below = analysis.analyse(below_mass, manifest, protocol, protocol_sha)
    assert below["components"]["query_marker@21/head-0"]["passes"] is False

    format_boundary = copy.deepcopy(rows)
    for index, row in enumerate(format_boundary):
        if index < 7:
            _replace_component(row, "query_marker@21", 0, 1.6, label_mass=0.1, format_ok=False)
    seven = analysis.analyse(format_boundary, manifest, protocol, protocol_sha)
    assert seven["components"]["query_marker@21/head-0"]["aggregate"]["format_rate"] >= 0.90
    assert seven["components"]["query_marker@21/head-0"]["passes"] is True
    _replace_component(
        format_boundary[7], "query_marker@21", 0, 1.6, label_mass=0.1, format_ok=False
    )
    eight = analysis.analyse(format_boundary, manifest, protocol, protocol_sha)
    assert eight["components"]["query_marker@21/head-0"]["aggregate"]["format_rate"] < 0.90
    assert eight["components"]["query_marker@21/head-0"]["passes"] is False


def test_sparse_selection_never_truncates_and_requires_both_families() -> None:
    root = Path(__file__).resolve().parents[1]
    rows, manifest, protocol, protocol_sha = _artifact(root)

    diffuse = copy.deepcopy(rows)
    for row in diffuse:
        for head in (1, 2, 3):
            _replace_component(row, "query_marker@21", head, 1.6)
    five = analysis.analyse(diffuse, manifest, protocol, protocol_sha)
    assert five["selection"]["n_qualifying_components"] == 5
    assert len(five["selection"]["qualifying_components"]) == 5
    assert five["gates"]["proceed_to_separately_frozen_stage1c"] is False
    assert "qualifying_component_count_outside_2_to_4" in five["gates"]["stop_reasons"]

    one_family = copy.deepcopy(rows)
    for row in one_family:
        _replace_component(row, "final_answer@26", 0, 2.0)
        _replace_component(row, "query_marker@21", 1, 1.6)
    missing = analysis.analyse(one_family, manifest, protocol, protocol_sha)
    assert missing["selection"]["n_qualifying_components"] == 2
    assert missing["selection"]["represented_receiver_families"] == ["query_marker"]
    assert missing["gates"]["proceed_to_separately_frozen_stage1c"] is False
    assert "missing_receiver_family:final_answer" in missing["gates"]["stop_reasons"]
