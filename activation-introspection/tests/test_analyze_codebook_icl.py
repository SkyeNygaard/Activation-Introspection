from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from introspect.codebook_icl import ANSWER_PREFIX, CONDITIONS, LABELS, exact_episodes, sha256_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_codebook_icl as analysis


def _score(predicted: str, correct: str, direction_hash: str | None) -> dict[str, Any]:
    conditional = {"Q": 0.8 if predicted == "Q" else 0.2, "K": 0.2 if predicted == "Q" else 0.8}
    label_mass = 0.9
    return {
        "direction_sha256": direction_hash,
        "predicted_label": predicted,
        "correct": predicted == correct,
        "conditional_probs": conditional,
        "full_logprobs": {label: math.log(label_mass * conditional[label]) for label in LABELS},
        "label_mass": label_mass,
        "format_ok": True,
    }


def _artifact(*, schema_version: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    concepts = ["mountain"]
    samples = ["A neutral record is visible. Hidden state marker: §"]
    directions = {
        "mountain": {
            "clean": None,
            "target": "target-hash",
            "random": "random-hash",
            "shuffled": "shuffled-hash",
            "test_only": "target-hash",
        }
    }
    config: dict[str, Any] = {
        "schema_version": schema_version,
        "concepts": concepts,
        "conditions": list(CONDITIONS),
        "labels": list(LABELS),
        "visible_samples": samples,
        "exact_cells_per_concept_carrier": 24,
        "direction_sha256": directions,
    }
    config_sha = analysis._json_sha256(config)
    rows: list[dict[str, Any]] = []
    correct_counts = {condition: 0 for condition in CONDITIONS}
    for episode in exact_episodes(samples[0]):
        prompt = f"<chat>{episode.render_user()}</chat>{ANSWER_PREFIX}"
        token_ids = list(prompt.encode())
        scores = {}
        for condition in CONDITIONS:
            predicted = episode.correct_label if condition == "target" else "Q"
            score = _score(predicted, episode.correct_label, directions["mountain"][condition])
            scores[condition] = score
            correct_counts[condition] += int(score["correct"])
        row: dict[str, Any] = {
            "schema_version": schema_version,
            "config_sha256": config_sha,
            "concept": "mountain",
            "carrier_id": 0,
            "cell_id": episode.cell_id,
            "episode_sha256": episode.digest(),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "state_token_positions": [1, 2, 3, 4, 5],
            "demo_signs": list(episode.demo_signs),
            "query_sign": episode.query_sign,
            "label_mapping": {
                "+1": episode.positive_label,
                "-1": episode.negative_label,
            },
            "correct_label": episode.correct_label,
            "condition_scores": scores,
        }
        if schema_version >= 2:
            row["token_ids"] = token_ids
            row["token_ids_sha256"] = analysis._json_sha256(token_ids)
        rows.append(row)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "config": config,
        "config_sha256": config_sha,
        "raw_sha256": "filled-when-written",
        "n_episode_rows": len(rows),
        "n_scored_forwards": len(rows) * len(CONDITIONS),
        "correct_by_condition": correct_counts,
    }
    return rows, manifest


def _write_artifact(
    directory: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[Path, Path]:
    raw_path = directory / "raw.jsonl"
    manifest_path = directory / "raw.manifest.json"
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    raw_path.write_text(raw)
    manifest["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    return raw_path, manifest_path


def test_load_verified_recomputes_manifest_config_hash(tmp_path: Path) -> None:
    rows, manifest = _artifact()
    raw_path, manifest_path = _write_artifact(tmp_path, rows, manifest)
    loaded, _ = analysis.load_verified(raw_path, manifest_path)
    assert len(loaded) == 24

    manifest["config"]["concepts"] = ["tampered"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config SHA-256"):
        analysis.load_verified(raw_path, manifest_path)


@pytest.mark.parametrize(
    "tamper",
    (
        "prompt",
        "prompt_leakage",
        "token_ids",
        "episode",
        "episode_hash",
        "direction",
        "prediction",
        "correctness",
        "probability",
    ),
)
def test_design_validation_rejects_tampered_rows(tamper: str) -> None:
    rows, manifest = _artifact()
    changed = copy.deepcopy(rows)
    row = changed[0]
    if tamper == "prompt":
        row["prompt"] += " QUERY SIGN: +1"
    elif tamper == "prompt_leakage":
        row["prompt"] += " QUERY SIGN: +1"
        row["prompt_sha256"] = sha256_text(row["prompt"])
    elif tamper == "token_ids":
        row["token_ids"][0] += 1
    elif tamper == "episode":
        row["demo_signs"] = [1, 1, 1, -1]
    elif tamper == "episode_hash":
        row["episode_sha256"] = "forged"
    elif tamper == "direction":
        row["condition_scores"]["target"]["direction_sha256"] = "forged"
    elif tamper == "prediction":
        row["condition_scores"]["target"]["predicted_label"] = "not-a-label"
    elif tamper == "correctness":
        row["condition_scores"]["target"]["correct"] = False
    elif tamper == "probability":
        row["condition_scores"]["target"]["conditional_probs"]["Q"] = 0.4
    with pytest.raises(ValueError):
        analysis.validate_design(changed, manifest)


def test_v2_requires_token_ids_but_v1_remains_readable() -> None:
    rows, manifest = _artifact()
    rows[0].pop("token_ids")
    with pytest.raises(ValueError, match="v2 row lacks token ids"):
        analysis.validate_design(rows, manifest)

    old_rows, old_manifest = _artifact(schema_version=1)
    assert analysis.validate_design(old_rows, old_manifest)["cells_per_cluster"] == 24


def test_exact_crossed_bootstrap_uses_all_multinomial_count_vectors() -> None:
    count_vectors, weights = analysis._resample_counts(2)
    observed = {
        tuple(vector): weight for vector, weight in zip(count_vectors, weights, strict=True)
    }
    assert observed == {(0.0, 2.0): 0.25, (1.0, 1.0): 0.5, (2.0, 0.0): 0.25}

    rows = []
    target = np.ones((2, 2))
    controls = {
        "left": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "right": np.array([[0.0, 0.0], [0.0, 1.0]]),
    }
    for concept_id, concept in enumerate(("a", "b")):
        for carrier in range(2):
            rows.append(
                {
                    "concept": concept,
                    "carrier_id": carrier,
                    "condition": "target",
                    "metric": target[concept_id, carrier],
                }
            )
            for condition, values in controls.items():
                rows.append(
                    {
                        "concept": concept,
                        "carrier_id": carrier,
                        "condition": condition,
                        "metric": values[concept_id, carrier],
                    }
                )

    point, lo, hi = analysis.exact_crossed_bootstrap(rows, "metric", "target", ("left", "right"))
    assert (point, lo, hi) == (0.75, 0.0, 1.0)


def test_analyse_reports_exact_not_monte_carlo_intervals() -> None:
    rows, manifest = _artifact()
    summary = analysis.analyse(rows, manifest)
    assert summary["analysis"] == {
        "bootstrap": "exact crossed concept x carrier multinomial resampling",
        "resample_count_pairs": 1,
    }
    assert summary["arms"]["target"]["correct"] == {
        "value": 1.0,
        "ci95": [1.0, 1.0],
    }
    assert summary["causal_icl_gate"] == {
        "target_accuracy_lower_gt_0.50": True,
        "target_minus_test_only_lower_gt_0.10": True,
        "target_format_at_least_0.90": True,
        "passed": True,
    }
