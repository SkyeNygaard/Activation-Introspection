"""Fail-closed analysis of the fixed-versus-remapped convention training study.

Three estimands, two of which have nulls fixed by the design rather than measured:

  row accuracy
      Ordinary per-episode accuracy. Chance is 0.500.

  query-twin pair accuracy
      Inside one episode the two query signs produce byte-identical prompts with
      opposite correct labels. A pair counts only if both members are right, so a
      learner reading only the prompt scores exactly 0.000. This rules out
      visible-text shortcuts.

  mapping-flip pair accuracy
      Two episodes sharing a demonstration order and query sign but using
      opposite label conventions receive identical hidden-state interventions and
      have opposite correct labels. A pair counts only if both are right, so a
      fixed sign-to-token probe scores exactly 0.000. This rules out the reading
      that a trained reporter is a sign detector wired to the output head.

Both nulls are identities. Neither is estimated from data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("base", "fixed", "remap")
CONDITIONS = ("target", "random")
TRAIN_STRENGTH = 0.5
MIN_FORMAT_RATE = 0.90
MIN_LABEL_MASS = 0.50
MIN_BASE_ACCURACY = 0.60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_seed(raw: Path, protocol_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(raw.with_suffix(".manifest.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    if manifest["raw_sha256"] != _sha256(raw):
        raise SystemExit(f"{raw} does not match its manifest hash")
    if manifest["config"]["protocol_sha256"] != _sha256(protocol_path):
        raise SystemExit(f"{raw} was produced under a different protocol")
    if _json_sha256(manifest["config"]) != manifest["config_sha256"]:
        raise SystemExit(f"{raw} manifest config hash does not verify")
    if manifest["config"]["source_files_sha256"] != protocol["source_files_sha256"]:
        raise SystemExit(f"{raw} sources differ from the frozen protocol")
    if manifest["config"].get("smoke"):
        raise SystemExit("refusing to analyse a smoke artifact")
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(rows) != manifest["n_rows"]:
        raise SystemExit(f"{raw} row count does not match its manifest")

    design = protocol["design"]
    strengths = design.get("eval_strengths", [TRAIN_STRENGTH])
    per_cell = (
        len(design["eval_concepts"])
        * len(design["eval_carriers"])
        * design["cells_per_concept_carrier"]
    )
    # Every condition is scored at the training strength; below it only the
    # concept direction is, so the row count is not a plain product.
    expected = len(ARMS) * per_cell * (len(CONDITIONS) + (len(strengths) - 1))
    if len(rows) != expected:
        raise SystemExit(f"{raw}: expected {expected} rows for the frozen design, got {len(rows)}")
    return rows, manifest


def _paired(rows: list[dict[str, Any]], key: tuple[str, ...], across: str) -> float:
    """Fraction of pairs in which both members are correct."""
    groups: dict[tuple[Any, ...], dict[Any, bool]] = defaultdict(dict)
    for row in rows:
        groups[tuple(row[field] for field in key)][row[across]] = bool(row["correct"])
    complete = [members for members in groups.values() if len(members) == 2]
    if not complete:
        raise SystemExit(f"no complete pairs across {across}")
    if len(complete) != len(groups):
        raise SystemExit(f"incomplete pairs across {across}")
    return sum(all(members.values()) for members in complete) / len(complete)


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "row_accuracy": sum(bool(r["correct"]) for r in rows) / len(rows),
        "query_twin_pair_accuracy": _paired(
            rows,
            ("arm", "condition", "concept", "carrier_sha256", "order_key", "positive_label"),
            "query_sign",
        ),
        "mapping_flip_pair_accuracy": _paired(
            rows,
            ("arm", "condition", "concept", "carrier_sha256", "order_key", "query_sign"),
            "positive_label",
        ),
        "format_rate": sum(bool(r["format_ok"]) for r in rows) / len(rows),
        "mean_label_mass": sum(float(r["label_mass"]) for r in rows) / len(rows),
        "mean_signed_margin": sum(float(r["signed_margin"]) for r in rows) / len(rows),
    }


def _spread(values: list[float]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("results/remap_training_protocol_v1.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    per_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for raw in args.raw:
        rows, manifest = load_seed(raw, args.protocol)
        seed = int(manifest["config"]["train_seed"])
        if seed in per_seed:
            raise SystemExit(f"duplicate seed {seed}")
        strengths = sorted({float(r.get("strength", TRAIN_STRENGTH)) for r in rows}, reverse=True)
        seed_cells: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            for condition in CONDITIONS:
                for strength in strengths:
                    selected = [
                        r
                        for r in rows
                        if r["arm"] == arm
                        and r["condition"] == condition
                        and float(r.get("strength", TRAIN_STRENGTH)) == strength
                    ]
                    if selected:
                        seed_cells[f"{arm}/{condition}@{strength:g}"] = _cell(selected)
        per_seed[seed] = seed_cells

    seeds = sorted(per_seed)
    cells = sorted(per_seed[seeds[0]])
    metrics = (
        "row_accuracy",
        "query_twin_pair_accuracy",
        "mapping_flip_pair_accuracy",
        "format_rate",
        "mean_label_mass",
    )
    pooled = {
        cell: {
            metric: _spread([per_seed[seed][cell][metric] for seed in seeds]) for metric in metrics
        }
        for cell in cells
    }

    def value(seed: int, cell: str, metric: str) -> float:
        return float(per_seed[seed][cell][metric])

    # Gates live in this file, not in the protocol JSON, so a later edit here
    # would silently re-judge an earlier artifact. Protocol v1 declared three
    # substantive gates and two of them failed; that verdict is a fact about v1
    # and must not be overwritten by gates written afterwards. The gate set is
    # therefore selected by which protocol produced the artifact.
    protocol = json.loads(args.protocol.read_text())
    declares_strengths = "eval_strengths" in protocol["design"]
    train = f"@{TRAIN_STRENGTH:g}"

    if not declares_strengths:
        gates_v1: dict[str, Any] = {
            "base_row_accuracy_above_0_60": all(
                value(s, f"base/target{train}", "row_accuracy") > MIN_BASE_ACCURACY for s in seeds
            ),
            "remap_beats_fixed_on_row_accuracy_every_seed": all(
                value(s, f"remap/target{train}", "row_accuracy")
                > value(s, f"fixed/target{train}", "row_accuracy")
                for s in seeds
            ),
            "fixed_loses_mapping_flip_against_base_every_seed": all(
                value(s, f"fixed/target{train}", "mapping_flip_pair_accuracy")
                < value(s, f"base/target{train}", "mapping_flip_pair_accuracy")
                for s in seeds
            ),
            "every_arm_verbalizes": all(
                value(s, cell, "format_rate") >= MIN_FORMAT_RATE
                and value(s, cell, "mean_label_mass") >= MIN_LABEL_MASS
                for s in seeds
                for cell in cells
            ),
        }
        return _emit(args, seeds, pooled, per_seed, cells, gates_v1, "v1", value)

    gates: dict[str, Any] = {
        "A_base_row_accuracy_above_0_60": all(
            value(s, f"base/target{train}", "row_accuracy") > MIN_BASE_ACCURACY for s in seeds
        ),
        "B_every_arm_verbalizes": all(
            value(s, cell, "format_rate") >= MIN_FORMAT_RATE
            and value(s, cell, "mean_label_mass") >= MIN_LABEL_MASS
            for s in seeds
            for cell in cells
        ),
        "D_generic_detection_trained_beats_base_on_random": all(
            value(s, f"{arm}/random{train}", "row_accuracy")
            > value(s, f"base/random{train}", "row_accuracy")
            for s in seeds
            for arm in ("fixed", "remap")
        ),
    }
    if "base/target@0.25" in per_seed[seeds[0]]:
        gates["C_transfer_trained_beats_base_at_0_25"] = all(
            value(s, f"{arm}/target@0.25", "row_accuracy")
            > value(s, "base/target@0.25", "row_accuracy")
            for s in seeds
            for arm in ("fixed", "remap")
        )
    return _emit(args, seeds, pooled, per_seed, cells, gates, "v2", value)


def _emit(
    args: argparse.Namespace,
    seeds: list[int],
    pooled: dict[str, Any],
    per_seed: dict[int, dict[str, Any]],
    cells: list[str],
    gates: dict[str, Any],
    gate_set: str,
    value: Any,
) -> None:
    summary = {
        "schema_version": 2,
        "status": "MULTI_SEED_DEV_EVIDENCE_NOT_A_POPULATION_INTERVAL",
        "analyzer_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(args.protocol),
        "gate_set": gate_set,
        "gate_set_note": (
            "gates are selected by the protocol that produced the artifact, so "
            "later analyzer versions cannot retroactively re-judge an earlier "
            "run. Protocol v1 declared a hypothesis that its own data falsified; "
            "that verdict is preserved rather than recomputed under v2 gates."
        ),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "structural_nulls": {
            "prompt_only_learner": {"row_accuracy": 0.5, "query_twin_pair_accuracy": 0.0},
            "fixed_sign_to_token_probe": {
                "row_accuracy": 0.5,
                "mapping_flip_pair_accuracy": 0.0,
            },
            "note": "identities of the design; neither is estimated from data",
        },
        "pooled": pooled,
        "per_seed": {str(seed): per_seed[seed] for seed in seeds},
        "contrasts": {
            f"{arm}_minus_base_{metric}_{cell.split('@')[1]}": _spread(
                [value(s, cell, metric) - value(s, base_cell, metric) for s in seeds]
            )
            for arm in ("fixed", "remap")
            for metric in ("row_accuracy", "mapping_flip_pair_accuracy")
            for cell, base_cell in (
                (c, c.replace(f"{arm}/", "base/", 1))
                for c in cells
                if c.startswith(f"{arm}/target@")
            )
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    temporary = args.out.with_name(f".{args.out.name}.tmp")
    created = False
    try:
        with temporary.open("x") as handle:
            created = True
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, args.out)
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise

    print(f"wrote {args.out}; {len(seeds)} seeds; gate set {gate_set}")
    for cell in cells:
        acc = pooled[cell]["row_accuracy"]
        flip = pooled[cell]["mapping_flip_pair_accuracy"]
        print(
            f"  {cell:18s} acc {acc['mean']:.3f} [{acc['min']:.3f},{acc['max']:.3f}]  "
            f"flip {flip['mean']:.3f} [{flip['min']:.3f},{flip['max']:.3f}]"
        )
    print(f"  all_gates_pass={summary['all_gates_pass']}")


if __name__ == "__main__":
    main()
