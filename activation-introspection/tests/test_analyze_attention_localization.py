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

import analyze_attention_localization as analysis


def _digest(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _score(
    correct_label: str,
    margin: float,
    tag: str,
    kl_key: str,
    *,
    label_mass: float = 0.9,
) -> dict[str, Any]:
    correct_probability = 1 / (1 + math.exp(-margin))
    probabilities = {
        correct_label: correct_probability,
        LABELS[1 - LABELS.index(correct_label)]: 1 - correct_probability,
    }
    logits = {
        correct_label: margin / 2,
        LABELS[1 - LABELS.index(correct_label)]: -margin / 2,
    }
    predicted = max(LABELS, key=logits.__getitem__)
    return {
        "predicted_label": predicted,
        "correct": predicted == correct_label,
        "signed_margin": margin,
        "conditional_correct_probability": correct_probability,
        "label_logits": logits,
        "full_logprobs": {label: math.log(label_mass * probabilities[label]) for label in LABELS},
        "label_mass": label_mass,
        "format_ok": True,
        "full_logits_sha256": _digest(tag),
        kl_key: 0.0 if "target_to_target" in kl_key else 0.1,
    }


def _artifact(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    protocol_path = root / "results/attention_localization_dev_protocol_v2.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    carrier = protocol["design"]["carrier"]
    all_episodes = exact_episodes(carrier)
    orders = sorted({episode.demo_signs for episode in all_episodes})
    episodes = [
        episode
        for order_id, order in enumerate(orders)
        for episode in all_episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[order_id % 2]
    ]
    rows: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    for episode in episodes:
        prompt = f"<chat>{episode.render_user()}</chat>Label:"
        token_ids = list(prompt.encode())
        state_positions = [match.start() for match in re.finditer("§", prompt)]
        label_positions = [match.start(1) for match in re.finditer(r"Label: ([QK])", prompt)]
        target = _score(
            episode.correct_label, 2.0, f"{episode.cell_id}/target", "target_to_target_kl_nats"
        )
        test_only = _score(
            episode.correct_label,
            0.0,
            f"{episode.cell_id}/test-only",
            "target_to_test_only_kl_nats",
        )
        patched: dict[str, dict[str, dict[str, Any]]] = {}
        for layer in analysis.LAYERS:
            patched[str(layer)] = {}
            for role in analysis.ROLES:
                margin = 2.0
                if role == "demo_labels" and layer == 10:
                    margin = 0.8
                elif role == "demo_labels" and layer == 11:
                    margin = 1.2
                elif role == "query_marker" and layer == 20:
                    margin = 1.5
                elif role == "all_positions" and layer == 30:
                    margin = 0.0
                patched[str(layer)][role] = _score(
                    episode.correct_label,
                    margin,
                    f"{episode.cell_id}/{layer}/{role}",
                    "target_to_patched_kl_nats",
                )
        context_layers = analysis.SENTINEL_LAYERS + analysis.LAYERS
        target_hashes = {
            str(layer): _digest(f"{episode.cell_id}/target/{layer}") for layer in context_layers
        }
        test_hashes = {
            str(layer): (
                target_hashes[str(layer)]
                if layer in analysis.SENTINEL_LAYERS
                else _digest(f"{episode.cell_id}/test/{layer}")
            )
            for layer in context_layers
        }
        row: dict[str, Any] = {
            "schema_version": 1,
            "cell_id": episode.cell_id,
            "episode_sha256": episode.digest(),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "token_ids": token_ids,
            "token_ids_sha256": analysis._json_sha256(token_ids),
            "state_token_positions": state_positions,
            "demo_label_positions": label_positions,
            "final_answer_position": len(token_ids) - 1,
            "demo_signs": list(episode.demo_signs),
            "query_sign": episode.query_sign,
            "label_mapping": {"+1": episode.positive_label, "-1": episode.negative_label},
            "correct_label": episode.correct_label,
            "baseline": {"target": target, "test_only": test_only},
            "instrumentation": {
                "pre_injection_sentinels_equal": True,
                "self_patch_score": {
                    **copy.deepcopy(target),
                    "target_to_self_patch_kl_nats": 0.0,
                },
                "self_patch_max_abs_logit_error": 0.0,
                "attention_context_sha256": {
                    "target": target_hashes,
                    "test_only": test_hashes,
                },
            },
            "patched": patched,
        }
        row["instrumentation"]["self_patch_score"].pop("target_to_target_kl_nats")
        rows.append(row)
        prompt_hashes.append(row["prompt_sha256"])

    config: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEV_ONLY_NOT_CONFIRMATORY",
        "estimand": "layerwise necessity for demonstration-mediated hidden-state reporting",
        "model_requested": "qwen-3b",
        "model_resolved": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": protocol["design"]["model_revision"],
        "device": "cpu",
        "dtype": "torch.float32",
        "concept": "ocean",
        "carrier": carrier,
        "injection_layer": 9,
        "strength": 1.0,
        "layers": list(analysis.LAYERS),
        "pre_injection_sentinel_layers": list(analysis.SENTINEL_LAYERS),
        "receiver_roles": list(analysis.ROLES),
        "patch": "all query-head contexts from the paired test_only donor before o_proj",
        "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
        "offline_model_loading": True,
        "self_patch_full_logit_tolerance": 1e-6,
        "n_query_heads": 16,
        "head_width": 128,
        "cell_ids": [episode.cell_id for episode in episodes],
        "direction_sha256": _digest("direction"),
        "centering_direction_sha256": _digest("center"),
        "prompt_set_sha256": analysis._json_sha256(sorted(prompt_hashes)),
        "max_offdiagonal_cosine": 0.2,
        "source_files_sha256": protocol["source_files_sha256"],
        "source_sha256": analysis._json_sha256(protocol["source_files_sha256"]),
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "smoke": False,
        "publishable": False,
    }
    config_sha = analysis._json_sha256(config)
    for row in rows:
        row["config_sha256"] = config_sha
    manifest = {
        "schema_version": 1,
        "config": config,
        "config_sha256": config_sha,
        "raw": "synthetic.jsonl",
        "raw_sha256": "",
        "n_episode_rows": 12,
        "n_scored_forwards": 12 * (3 + len(analysis.LAYERS) * len(analysis.ROLES)),
    }
    return rows, manifest, protocol, protocol_sha


def test_frozen_analysis_validates_grid_selects_syntax_layers_and_rejects_tampering(
    tmp_path: Path,
) -> None:
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
        root / "results/attention_localization_dev_protocol_v2.json",
    )
    summary = analysis.analyse(loaded, loaded_manifest, loaded_protocol, loaded_sha)

    assert loaded_protocol == protocol and loaded_sha == protocol_sha
    assert summary["design_validation"] == {"cells": 12, "layers": 26, "roles": 4, "patches": 1248}
    assert summary["selection"] == {
        "demo_labels": [10, 11],
        "query_marker": [20],
        "final_answer": [],
    }
    assert summary["gates"]["proceed_to_separately_frozen_head_screen"] is True
    assert summary["layer_role"]["demo_labels"]["10"][
        "aggregate_margin_removal_fraction"
    ] == pytest.approx(0.6)
    assert summary["layer_role"]["all_positions"]["30"]["passes_candidate_gate"] is True
    assert (
        summary["analyzer_sha256"]
        == hashlib.sha256(
            (root / "scripts/analyze_attention_localization.py").read_bytes()
        ).hexdigest()
    )
    summary_path = tmp_path / "summary.json"
    analysis._write_summary(summary_path, summary)
    assert json.loads(summary_path.read_text()) == summary
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analysis._write_summary(summary_path, summary)

    envelope_only = copy.deepcopy(rows)
    for row in envelope_only:
        for role, layer in (("demo_labels", "10"), ("demo_labels", "11"), ("query_marker", "20")):
            row["patched"][layer][role] = _score(
                row["correct_label"],
                2.0,
                f"envelope-only/{row['cell_id']}/{layer}/{role}",
                "target_to_patched_kl_nats",
            )
    stopped = analysis.analyse(envelope_only, manifest, protocol, protocol_sha)
    assert stopped["selection"] == {
        "demo_labels": [],
        "query_marker": [],
        "final_answer": [],
    }
    assert stopped["gates"]["only_all_positions_passes"] is True
    assert stopped["gates"]["stop_reason"] == "only_all_positions_passed"

    negative_denominator = copy.deepcopy(rows)
    negative_target = _score(
        negative_denominator[0]["correct_label"],
        -100.0,
        "negative-denominator/target",
        "target_to_target_kl_nats",
    )
    negative_denominator[0]["baseline"]["target"] = negative_target
    negative_denominator[0]["instrumentation"]["self_patch_score"] = {
        **copy.deepcopy(negative_target),
        "target_to_self_patch_kl_nats": 0.0,
    }
    negative_denominator[0]["instrumentation"]["self_patch_score"].pop("target_to_target_kl_nats")
    no_inverted_selection = analysis.analyse(negative_denominator, manifest, protocol, protocol_sha)
    assert no_inverted_selection["gates"]["baseline_gate_passes"] is True
    assert no_inverted_selection["gates"]["positive_baseline_margin_gap"] is False
    assert all(
        candidate["aggregate_margin_removal_fraction"] is None
        for role in analysis.ROLES
        for candidate in no_inverted_selection["layer_role"][role].values()
    )

    broken_sham = copy.deepcopy(rows)
    broken_sham[0]["instrumentation"]["self_patch_score"]["full_logits_sha256"] = _digest(
        "wrong-self-logits"
    )
    with pytest.raises(ValueError, match="self-patch full logits"):
        analysis.validate_design(broken_sham, manifest, protocol)

    unprovable_format = copy.deepcopy(rows)
    unprovable_format[0]["patched"]["10"]["demo_labels"] = _score(
        unprovable_format[0]["correct_label"],
        2.0,
        "unprovable-format",
        "target_to_patched_kl_nats",
        label_mass=0.4,
    )
    with pytest.raises(ValueError, match="format_ok=True cannot be certified"):
        analysis.validate_design(unprovable_format, manifest, protocol)

    broken = copy.deepcopy(rows)
    del broken[0]["patched"]["10"]["query_marker"]
    with pytest.raises(ValueError, match="receiver-role grid"):
        analysis.validate_design(broken, manifest, protocol)

    raw_path.write_text(raw + " ")
    with pytest.raises(ValueError, match="raw artifact SHA-256"):
        analysis.load_verified(
            raw_path,
            manifest_path,
            root / "results/attention_localization_dev_protocol_v2.json",
        )
