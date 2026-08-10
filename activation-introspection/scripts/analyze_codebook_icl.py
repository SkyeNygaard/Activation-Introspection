"""Verify and analyse a saved opaque-codebook ICL run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect.codebook_icl import ANSWER_PREFIX, CONDITIONS, LABELS, exact_episodes, sha256_text

CONTROLS = ("random", "shuffled", "test_only")
_FLOAT_TOLERANCE = 2e-5


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_verified(
    raw_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    config_sha = manifest["config_sha256"]
    if _json_sha256(manifest["config"]) != config_sha:
        raise ValueError("manifest config SHA-256 does not match its config")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_sha != manifest["raw_sha256"]:
        raise ValueError("raw SHA-256 does not match manifest")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if len(rows) != manifest["n_episode_rows"]:
        raise ValueError("raw row count does not match manifest")
    if any(row["config_sha256"] != config_sha for row in rows):
        raise ValueError("a raw row refers to a different config")
    return rows, manifest


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} is not a finite probability")
    return number


def _validate_score(
    score: dict[str, Any],
    *,
    concept: str,
    condition: str,
    correct_label: str,
    config: dict[str, Any],
) -> None:
    location = f"{concept}/{condition}"
    if score["direction_sha256"] != config["direction_sha256"][concept][condition]:
        raise ValueError(f"direction hash drifted in {location}")
    if not isinstance(score["correct"], bool) or not isinstance(score["format_ok"], bool):
        raise ValueError(f"non-boolean score flag in {location}")

    conditional = score["conditional_probs"]
    full_logprobs = score["full_logprobs"]
    if set(conditional) != set(LABELS) or set(full_logprobs) != set(LABELS):
        raise ValueError(f"label probability keys drifted in {location}")
    probs = {label: _probability(conditional[label], f"{location}/{label}") for label in LABELS}
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"conditional probabilities do not sum to one in {location}")

    logprobs: dict[str, float] = {}
    for label in LABELS:
        value = full_logprobs[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"full log-probability is not numeric in {location}")
        logprob = float(value)
        if not math.isfinite(logprob) or logprob > _FLOAT_TOLERANCE:
            raise ValueError(f"invalid full log-probability in {location}")
        logprobs[label] = logprob

    derived_mass = sum(math.exp(logprobs[label]) for label in LABELS)
    if derived_mass <= 0:
        raise ValueError(f"label mass underflowed in {location}")
    label_mass = _probability(score["label_mass"], f"{location}/label_mass")
    if not math.isclose(derived_mass, label_mass, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(f"label mass is inconsistent in {location}")
    for label in LABELS:
        derived = math.exp(logprobs[label]) / derived_mass
        if not math.isclose(derived, probs[label], abs_tol=_FLOAT_TOLERANCE):
            raise ValueError(f"conditional probability is inconsistent in {location}")

    predicted = max(LABELS, key=probs.__getitem__)
    if score["predicted_label"] != predicted:
        raise ValueError(f"predicted label is inconsistent in {location}")
    if score["correct"] != (predicted == correct_label):
        raise ValueError(f"correctness flag is inconsistent in {location}")


def validate_design(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, int]:
    """Fail closed if any stored design or score invariant drifted."""
    config = manifest["config"]
    if _json_sha256(config) != manifest["config_sha256"]:
        raise ValueError("manifest config SHA-256 does not match its config")
    if config["conditions"] != list(CONDITIONS) or config["labels"] != list(LABELS):
        raise ValueError("configured conditions or labels drifted")
    concepts = config["concepts"]
    samples = config["visible_samples"]
    if len(concepts) != len(set(concepts)) or len(samples) != len(set(samples)):
        raise ValueError("concepts and visible samples must be unique")
    if int(config["exact_cells_per_concept_carrier"]) != len(exact_episodes(samples[0])):
        raise ValueError("configured exact-cell count drifted")
    direction_hashes = config["direction_sha256"]
    if set(direction_hashes) != set(concepts) or any(
        set(direction_hashes[concept]) != set(CONDITIONS) for concept in concepts
    ):
        raise ValueError("direction hash grid drifted")
    if manifest["n_episode_rows"] != len(rows):
        raise ValueError("manifest episode-row count drifted")

    schema_version = int(manifest.get("schema_version", 1))
    if int(config.get("schema_version", schema_version)) != schema_version:
        raise ValueError("manifest and config schema versions differ")
    by_cluster: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    wrappers: set[tuple[str, str]] = set()
    correct_counts = {condition: 0 for condition in CONDITIONS}
    for row in rows:
        if row["config_sha256"] != manifest["config_sha256"]:
            raise ValueError("a raw row refers to a different config")
        concept = row["concept"]
        carrier_id = row["carrier_id"]
        if (
            concept not in concepts
            or not isinstance(carrier_id, int)
            or isinstance(carrier_id, bool)
        ):
            raise ValueError("row refers to an unknown concept or carrier")
        if not 0 <= carrier_id < len(samples):
            raise ValueError(f"carrier id out of range in {concept}/{row['cell_id']}")
        if int(row.get("schema_version", 1)) != schema_version:
            raise ValueError(f"schema version drifted in {concept}/{row['cell_id']}")
        by_cluster[(concept, carrier_id)].append(row)
        if set(row["condition_scores"]) != set(CONDITIONS):
            raise ValueError(f"incomplete conditions in {concept}/{row['cell_id']}")

        episodes = {episode.cell_id: episode for episode in exact_episodes(samples[carrier_id])}
        if row["cell_id"] not in episodes:
            raise ValueError(f"unknown episode cell in {concept}/{row['cell_id']}")
        episode = episodes[row["cell_id"]]
        expected_mapping = {"+1": episode.positive_label, "-1": episode.negative_label}
        if (
            tuple(row["demo_signs"]) != episode.demo_signs
            or row["query_sign"] != episode.query_sign
            or row["label_mapping"] != expected_mapping
            or row["correct_label"] != episode.correct_label
            or row["episode_sha256"] != episode.digest()
        ):
            raise ValueError(f"episode fields drifted in {concept}/{row['cell_id']}")

        prompt = row["prompt"]
        if not isinstance(prompt, str) or sha256_text(prompt) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash drifted in {concept}/{row['cell_id']}")
        rendered = episode.render_user()
        if prompt.count(rendered) != 1:
            raise ValueError(
                f"prompt does not contain the exact episode in {concept}/{row['cell_id']}"
            )
        prefix, suffix = prompt.split(rendered)
        wrappers.add((prefix, suffix))

        token_ids = row.get("token_ids")
        if token_ids is None and schema_version >= 2:
            raise ValueError(f"v2 row lacks token ids in {concept}/{row['cell_id']}")
        if token_ids is not None:
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(
                    not isinstance(token, int) or isinstance(token, bool) or token < 0
                    for token in token_ids
                )
                or _json_sha256(token_ids) != row["token_ids_sha256"]
            ):
                raise ValueError(f"token ids drifted in {concept}/{row['cell_id']}")
        positions = row["state_token_positions"]
        if (
            not isinstance(positions, list)
            or len(positions) != len(episode.state_signs)
            or positions != sorted(set(positions))
            or any(not isinstance(position, int) or position < 0 for position in positions)
            or (token_ids is not None and positions[-1] >= len(token_ids))
        ):
            raise ValueError(f"state token positions drifted in {concept}/{row['cell_id']}")

        for condition, score in row["condition_scores"].items():
            _validate_score(
                score,
                concept=concept,
                condition=condition,
                correct_label=row["correct_label"],
                config=config,
            )
            correct_counts[condition] += int(score["correct"])

    expected_clusters = {
        (concept, carrier) for concept in concepts for carrier in range(len(samples))
    }
    if set(by_cluster) != expected_clusters:
        raise ValueError("concept x carrier grid is incomplete")
    if len(wrappers) != 1 or not next(iter(wrappers))[1].endswith(ANSWER_PREFIX):
        raise ValueError("chat wrapper drifted across episodes")
    if "correct_by_condition" in manifest and manifest["correct_by_condition"] != correct_counts:
        raise ValueError("manifest correctness totals drifted")
    if "n_scored_forwards" in manifest and manifest["n_scored_forwards"] != len(rows) * len(
        CONDITIONS
    ):
        raise ValueError("manifest scored-forward count drifted")

    for (concept, carrier_id), cluster in by_cluster.items():
        expected = {episode.cell_id for episode in exact_episodes(samples[carrier_id])}
        found = {row["cell_id"] for row in cluster}
        if len(cluster) != len(expected) or found != expected:
            raise ValueError(f"exact cells are incomplete for {concept}/carrier-{carrier_id}")
        if [sum(row["correct_label"] == label for row in cluster) for label in ("Q", "K")] != [
            12,
            12,
        ]:
            raise ValueError(f"labels are unbalanced for {concept}/carrier-{carrier_id}")

        twins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cluster:
            twins[row["cell_id"].rsplit("q", 1)[0]].append(row)
        if len(twins) != 12:
            raise ValueError(f"query twins are incomplete for {concept}/carrier-{carrier_id}")
        for pair in twins.values():
            if (
                len(pair) != 2
                or {row["query_sign"] for row in pair} != {-1, 1}
                or len({row["prompt"] for row in pair}) != 1
                or (schema_version >= 2 and len({tuple(row["token_ids"]) for row in pair}) != 1)
                or {row["correct_label"] for row in pair} != {"Q", "K"}
            ):
                raise ValueError(f"query twins drifted for {concept}/carrier-{carrier_id}")
    return {
        "concepts": len(concepts),
        "carriers": len(samples),
        "cells_per_cluster": int(config["exact_cells_per_concept_carrier"]),
        "complete_conditions": len(CONDITIONS),
    }


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        for condition, score in row["condition_scores"].items():
            flat.append(
                {
                    "concept": row["concept"],
                    "carrier_id": row["carrier_id"],
                    "cell_id": row["cell_id"],
                    "query_sign": row["query_sign"],
                    "condition": condition,
                    "predicted_label": score["predicted_label"],
                    "correct": float(score["correct"]),
                    "p_correct": float(score["conditional_probs"][row["correct_label"]]),
                    "format_ok": float(score["format_ok"]),
                    "label_mass": float(score["label_mass"]),
                }
            )
    return flat


def query_twin_rows(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        pair_id = row["cell_id"].rsplit("q", 1)[0]
        pairs[(row["concept"], row["carrier_id"], pair_id, row["condition"])].append(row)
    out = []
    for (concept, carrier_id, pair_id, condition), pair in pairs.items():
        if len(pair) != 2 or {row["query_sign"] for row in pair} != {-1, 1}:
            raise ValueError(f"invalid query twin {concept}/{carrier_id}/{pair_id}/{condition}")
        out.append(
            {
                "concept": concept,
                "carrier_id": carrier_id,
                "condition": condition,
                "both_correct": float(all(row["correct"] for row in pair)),
                "prediction_flipped": float(len({row["predicted_label"] for row in pair}) == 2),
            }
        )
    return out


@cache
def _resample_counts(n_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Every multinomial bootstrap count vector and its exact probability."""
    if n_units < 1:
        raise ValueError("crossed bootstrap needs at least one unit on each axis")
    count_vectors = []
    probabilities = []
    numerator = math.factorial(n_units)
    for bars in combinations(range(2 * n_units - 1), n_units - 1):
        cuts = (-1, *bars, 2 * n_units - 1)
        counts = tuple(cuts[i + 1] - cuts[i] - 1 for i in range(n_units))
        count_vectors.append(counts)
        probabilities.append(
            numerator / math.prod(math.factorial(count) for count in counts) / n_units**n_units
        )
    return np.asarray(count_vectors, dtype=float), np.asarray(probabilities, dtype=float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, axis=None)
    ordered = values.ravel()[order]
    cumulative = np.cumsum(weights.ravel()[order])
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(ordered) - 1)
    return float(ordered[index])


def exact_crossed_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    condition: str,
    controls: tuple[str, ...] = (),
) -> tuple[float, float, float]:
    """Exact percentile interval over independent concept/carrier bootstrap counts."""
    if not rows:
        raise ValueError("crossed bootstrap needs at least one row")
    concepts = sorted({str(row["concept"]) for row in rows})
    carriers = sorted({int(row["carrier_id"]) for row in rows})
    conditions = (condition, *controls)
    matrices: dict[str, np.ndarray] = {}
    for arm in conditions:
        matrix = np.empty((len(concepts), len(carriers)), dtype=float)
        for concept_id, concept in enumerate(concepts):
            for carrier_id, carrier in enumerate(carriers):
                cluster_values = [
                    float(row[metric])
                    for row in rows
                    if row["concept"] == concept
                    and row["carrier_id"] == carrier
                    and row["condition"] == arm
                ]
                if not cluster_values:
                    raise ValueError(f"missing {arm} rows for {concept}/carrier-{carrier}")
                matrix[concept_id, carrier_id] = float(np.mean(cluster_values))
        matrices[arm] = matrix

    concept_counts, concept_weights = _resample_counts(len(concepts))
    carrier_counts, carrier_weights = _resample_counts(len(carriers))
    denominator = len(concepts) * len(carriers)

    def distribution(arm: str) -> np.ndarray:
        return np.asarray(
            np.einsum("ic,ck,jk->ij", concept_counts, matrices[arm], carrier_counts) / denominator
        )

    draws = distribution(condition)
    point = float(np.mean(matrices[condition]))
    if controls:
        draws = draws - np.maximum.reduce([distribution(control) for control in controls])
        point -= max(float(np.mean(matrices[control])) for control in controls)
    weights = concept_weights[:, None] * carrier_weights[None, :]
    return (
        point,
        _weighted_quantile(draws, weights, 0.025),
        _weighted_quantile(draws, weights, 0.975),
    )


def analyse(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    validation = validate_design(rows, manifest)
    flat = flatten(rows)
    twins = query_twin_rows(flat)
    summary: dict[str, Any] = {
        "raw_sha256": manifest["raw_sha256"],
        "config_sha256": manifest["config_sha256"],
        "n_episode_rows": len(rows),
        "design_validation": validation,
        "analysis": {
            "bootstrap": "exact crossed concept x carrier multinomial resampling",
            "resample_count_pairs": int(
                len(_resample_counts(validation["concepts"])[0])
                * len(_resample_counts(validation["carriers"])[0])
            ),
        },
        "arms": {},
        "contrasts": {},
        "query_twins": {},
    }
    for condition in CONDITIONS:
        summary["arms"][condition] = {}
        for metric in ("correct", "p_correct", "format_ok", "label_mass"):
            point, lo, hi = exact_crossed_bootstrap(flat, metric, condition)
            summary["arms"][condition][metric] = {"value": point, "ci95": [lo, hi]}
        summary["query_twins"][condition] = {}
        for metric in ("both_correct", "prediction_flipped"):
            point, lo, hi = exact_crossed_bootstrap(twins, metric, condition)
            summary["query_twins"][condition][metric] = {"value": point, "ci95": [lo, hi]}

    for control in (*CONTROLS, "clean"):
        summary["contrasts"][f"target_minus_{control}"] = {}
        for metric in ("correct", "p_correct"):
            point, lo, hi = exact_crossed_bootstrap(flat, metric, "target", (control,))
            summary["contrasts"][f"target_minus_{control}"][metric] = {
                "value": point,
                "ci95": [lo, hi],
            }
    for metric in ("correct", "p_correct"):
        point, lo, hi = exact_crossed_bootstrap(flat, metric, "target", CONTROLS)
        summary["contrasts"]["target_minus_strongest_control"] = summary["contrasts"].get(
            "target_minus_strongest_control", {}
        )
        summary["contrasts"]["target_minus_strongest_control"][metric] = {
            "value": point,
            "ci95": [lo, hi],
        }
    gates = {
        "target_accuracy_lower_gt_0.50": summary["arms"]["target"]["correct"]["ci95"][0] > 0.50,
        "target_minus_test_only_lower_gt_0.10": summary["contrasts"]["target_minus_test_only"][
            "correct"
        ]["ci95"][0]
        > 0.10,
        "target_format_at_least_0.90": summary["arms"]["target"]["format_ok"]["value"] >= 0.90,
    }
    summary["causal_icl_gate"] = {**gates, "passed": all(gates.values())}
    return summary


def plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ("clean", "test_only", "random", "shuffled", "target")
    labels = (
        "no hidden\nstate",
        "query\nonly",
        "random\ndirection",
        "shuffled\ndirection",
        "concept\ndirection",
    )
    values = [summary["arms"][arm]["correct"]["value"] for arm in order]
    cis = [summary["arms"][arm]["correct"]["ci95"] for arm in order]
    errors = np.array(
        [[value - ci[0], ci[1] - value] for value, ci in zip(values, cis, strict=True)]
    ).T

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    colors = ["#a8a8a8", "#777777", "#e6a34a", "#ce7b45", "#2b6f9f"]
    ax.bar(range(len(order)), values, yerr=errors, capsize=4, color=colors, width=0.72)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.2)
    ax.text(-0.45, 0.51, "balanced chance", fontsize=8, va="bottom")
    ax.set_xticks(range(len(order)), labels)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("opaque-label accuracy")
    ax.set_title("A model learns a hidden-state codebook from identical visible text")
    ax.text(
        0.5,
        -0.19,
        "95% exact crossed bootstrap over 8 confirm concepts x 3 fixed carrier prompts",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.raw.with_suffix(".manifest.json")
    rows, manifest = load_verified(args.raw, manifest_path)
    summary = analyse(rows, manifest)
    summary["analysis"]["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.figure:
        plot(summary, args.figure)


if __name__ == "__main__":
    main()
