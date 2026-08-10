"""Fail-closed analysis of the zero-demonstration reporter run.

Written against the frozen protocol text before the artifact existed. The gates
are evaluated here rather than in prose, so the published verdict is machine
output.

The estimand is twin-pair accuracy: a concept-carrier cell counts as correct only
when both byte-identical members of the twin receive their opposite labels. Per-
row accuracy is deliberately not the headline, because a model that always
answers ``Q`` scores 0.500 per row while scoring 0.000 on pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

CONDITIONS = ("target", "random", "shuffled")
ARMS = ("base", "trained", "trained_seen_bank")
CHANCE = 0.5

#: A discriminative readout over tokens the model would never emit is not a
#: verbalization. V1 ordered Q against K correctly at 0.917 twin-pair accuracy
#: while holding ~5e-9 total probability on the two labels; these thresholds
#: exist so that failure is a machine verdict rather than a footnote.
MIN_FORMAT_RATE = 0.90
MIN_LABEL_MASS = 0.50


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_verified(
    raw: Path, manifest_path: Path, protocol_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = _sha256(protocol_path)

    if manifest["raw_sha256"] != _sha256(raw):
        raise SystemExit("raw file does not match its manifest hash")
    if manifest["config"]["protocol_sha256"] != protocol_sha:
        raise SystemExit("manifest was produced under a different protocol")
    if _json_sha256(manifest["config"]) != manifest["config_sha256"]:
        raise SystemExit("manifest config hash does not verify")
    if manifest["config"]["source_sha256"] != _json_sha256(
        manifest["config"]["source_files_sha256"]
    ):
        raise SystemExit("source hash tree does not verify")
    if manifest["config"]["source_files_sha256"] != protocol["source_files_sha256"]:
        raise SystemExit("run sources differ from the frozen protocol sources")
    if manifest["config"].get("smoke"):
        raise SystemExit("refusing to analyse a smoke artifact")

    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(rows) != manifest["n_rows"]:
        raise SystemExit("row count does not match the manifest")

    design = protocol["design"]
    expected = (
        len(design["eval_concepts"])
        * len(design["eval_carriers"])
        * (len(CONDITIONS) * 2 + 1)
        * len(ARMS)
    )
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows for the frozen design, found {len(rows)}")
    return rows, manifest, protocol, protocol_sha


def _pairs(rows: list[dict[str, Any]], arm: str, condition: str) -> dict[str, dict[int, Any]]:
    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for row in rows:
        if row["arm"] != arm or row["condition"] != condition or row["sign"] == 0:
            continue
        out[f"{row['concept']}/{row['carrier_sha256'][:8]}"][row["sign"]] = row
    return out


def _arm_condition(rows: list[dict[str, Any]], arm: str, condition: str) -> dict[str, Any]:
    pairs = _pairs(rows, arm, condition)
    incomplete = [key for key, members in pairs.items() if set(members) != {-1, 1}]
    if incomplete:
        raise SystemExit(f"incomplete twins for {arm}/{condition}: {incomplete}")

    both, per_row, margins, masses, formats = [], [], [], [], []
    by_concept: dict[str, list[bool]] = defaultdict(list)
    for key, members in sorted(pairs.items()):
        correct = [bool(members[sign]["correct"]) for sign in (-1, 1)]
        both.append(all(correct))
        per_row.extend(correct)
        by_concept[key.split("/")[0]].append(all(correct))
        for sign in (-1, 1):
            margins.append(float(members[sign]["signed_margin"]))
            masses.append(float(members[sign]["label_mass"]))
            formats.append(bool(members[sign]["format_ok"]))

    return {
        "n_pairs": len(both),
        "twin_pair_accuracy": sum(both) / len(both),
        "row_accuracy": sum(per_row) / len(per_row),
        "mean_signed_margin": sum(margins) / len(margins),
        "mean_label_mass": sum(masses) / len(masses),
        "format_rate": sum(formats) / len(formats),
        "positive_concepts": sum(
            1 for values in by_concept.values() if sum(values) / len(values) > CHANCE
        ),
        "n_concepts": len(by_concept),
        "twin_pair_accuracy_by_concept": {
            name: sum(values) / len(values) for name, values in sorted(by_concept.items())
        },
    }


def _clean(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and row["condition"] == "clean"]
    if not selected:
        raise SystemExit(f"no clean rows for {arm}")
    predictions = [row["predicted_label"] for row in selected]
    majority = max(set(predictions), key=predictions.count)
    return {
        "n_rows": len(selected),
        "majority_label": majority,
        "majority_share": predictions.count(majority) / len(predictions),
        "mean_label_mass": sum(float(row["label_mass"]) for row in selected) / len(selected),
        "format_rate": sum(bool(row["format_ok"]) for row in selected) / len(selected),
    }


def analyse(
    rows: list[dict[str, Any]], manifest: dict[str, Any], protocol: dict[str, Any], sha: str
) -> dict[str, Any]:
    arms = {
        arm: {condition: _arm_condition(rows, arm, condition) for condition in CONDITIONS}
        for arm in ARMS
    }
    clean = {arm: _clean(rows, arm) for arm in ARMS}

    trained_target = arms["trained"]["target"]["twin_pair_accuracy"]
    base_target = arms["base"]["target"]["twin_pair_accuracy"]
    strongest_control = max(
        arms["trained"][condition]["twin_pair_accuracy"] for condition in ("random", "shuffled")
    )

    trained_format = arms["trained"]["target"]["format_rate"]
    trained_mass = arms["trained"]["target"]["mean_label_mass"]
    gates = {
        "trained_target_above_chance": trained_target > CHANCE,
        "untrained_base_at_or_below_chance": base_target <= CHANCE,
        "trained_target_above_strongest_control": trained_target > strongest_control,
        "trained_output_is_a_verbalization": (
            trained_format >= MIN_FORMAT_RATE and trained_mass >= MIN_LABEL_MASS
        ),
    }
    generic_readout = strongest_control > CHANCE

    return {
        "schema_version": 1,
        "status": "SINGLE_TRAINING_RUN_ONE_SEED_NOT_CONFIRMATORY",
        "protocol_sha256": sha,
        "raw_sha256": manifest["raw_sha256"],
        "config_sha256": manifest["config_sha256"],
        "analyzer_sha256": _sha256(Path(__file__)),
        "n_rows": len(rows),
        "n_train_steps": manifest["config"]["n_train_steps"],
        "final_epoch_mean_loss": manifest["config"]["final_epoch_mean_loss"],
        "chance_is_structural": (
            "twin members are byte-identical with opposite correct labels. A "
            "prompt-only strategy must answer both members the same way, so it "
            "scores exactly 0.500 per row and exactly 0.000 on pairs; a "
            "row-independent coin flip scores 0.250 on pairs. The 0.500 pair "
            "threshold in the gate below is therefore well above the null and "
            "is conservative, not the null itself."
        ),
        "arms": arms,
        "clean": clean,
        "contrasts": {
            "trained_minus_base_target": trained_target - base_target,
            "trained_target_minus_strongest_control": trained_target - strongest_control,
            "seen_minus_heldout_bank_target": (
                arms["trained_seen_bank"]["target"]["twin_pair_accuracy"] - trained_target
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "generic_perturbation_sign_readout": generic_readout,
        "interpretation_rule": protocol["stop"],
        "claim_boundary": protocol["claim_boundary"],
    }


def _write(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
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
        "--protocol", type=Path, default=Path("results/report_training_protocol_v1.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest or args.raw.with_suffix(".manifest.json")
    rows, manifest, protocol, sha = load_verified(args.raw, manifest_path, args.protocol)
    summary = analyse(rows, manifest, protocol, sha)
    _write(args.out, summary)
    print(
        f"wrote {args.out}; "
        f"trained target {summary['arms']['trained']['target']['twin_pair_accuracy']:.3f}; "
        f"base {summary['arms']['base']['target']['twin_pair_accuracy']:.3f}; "
        f"gates_pass={summary['all_gates_pass']}"
    )


if __name__ == "__main__":
    main()
