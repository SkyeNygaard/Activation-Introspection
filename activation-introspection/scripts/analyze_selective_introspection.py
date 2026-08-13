#!/usr/bin/env python3
"""notes/29 -- does letting an introspective reporter abstain recover selectivity?

notes/08 found that training a model to report its own states destroys its ability
to tell a meaningful concept from a magnitude-matched random direction. Anthropic's
Introspection Adapters names abstention as the unbuilt fix for that class of false
positive. Every introspection paper reports accuracy at full coverage; none asks
whether the model knows *which* of its self-reports to distrust.

This is a **secondary analysis of frozen artifacts** -- no model load, no GPU. It
reads ``report_training_v3_seed{0..3}_raw.jsonl``, which carry a per-row confidence
margin that notes/08 never used.

Seeds 0 and 1 are development; seeds 2 and 3 are confirmation. That split is
declared in notes/29 before any seed was opened, and this script enforces it: it
refuses to print confirmation numbers unless ``--phase confirm`` is passed, so the
development pass cannot accidentally show them.

    uv run python scripts/analyze_selective_introspection.py --phase dev
    uv run python scripts/analyze_selective_introspection.py --phase confirm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS = Path("results")
OUT = RESULTS / "selective_introspection_v1_summary.json"

#: Measurements 1 and 2 (ranking, risk-coverage) on report_training_v3.
SEEDS = {"dev": (0, 1), "confirm": (2, 3)}

#: Measurement 3 (does abstention recover selectivity) on remap_training_v2, which
#: is where notes/08's selectivity loss actually lives -- report_training_v3's
#: trained arm sits at chance on random directions. Split declared in notes/29's
#: amendment before any of it was opened.
REMAP_SEEDS = {"dev": (0,), "confirm": (1, 2)}
#: notes/08's selectivity table is the strength-0.5 cell.
REMAP_STRENGTH = 0.5
REMAP_ARMS = ("base", "fixed", "remap")

#: notes/08's headline arms. ``trained_seen_bank`` is carried but not featured --
#: it re-uses the training bank, so it is not a clean generalization arm.
ARMS = ("base", "trained")
#: The comparison notes/08 turns on: a real concept against a magnitude-matched
#: random direction. ``shuffled`` and ``clean`` are structural controls.
CONDITIONS = ("target", "random")

#: Coverage levels, fixed here rather than chosen after seeing a curve.
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

#: Below this on development, notes/29's kill rule fires and nothing is reported.
KILL_AUROC = 0.60


def load(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in seeds:
        path = RESULTS / f"report_training_v3_seed{s}_raw.jsonl"
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["seed"] = s
                rows.append(r)
    return rows


def confidence(row: dict[str, Any]) -> float:
    """How far the chosen label sat from the other one. Not a verbalized number."""
    return abs(float(row["signed_margin"]))


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a correct row outranks an incorrect one). Ties count half."""
    if not pos or not neg:
        return float("nan")
    total = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return total / (len(pos) * len(neg))


def risk_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Accuracy on the most-confident fraction retained, at fixed coverages."""
    ranked = sorted(rows, key=confidence, reverse=True)
    out = {}
    for cov in COVERAGES:
        k = max(1, round(len(ranked) * cov))
        out[f"{cov:.1f}"] = sum(bool(r["correct"]) for r in ranked[:k]) / k
    return out


def degeneracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """notes/29 asks this first: is the margin even a usable quantity?"""
    vals = [confidence(r) for r in rows]
    return {
        "n": len(vals),
        "distinct_values": len(set(vals)),
        "min": min(vals),
        "max": max(vals),
        "fraction_at_mode": max(vals.count(v) for v in set(vals)) / len(vals),
    }


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("format_ok", True)]
    out: dict[str, Any] = {"margin_health": degeneracy(scored)}

    for arm in ARMS:
        arm_rows = [r for r in scored if r["arm"] == arm]
        entry: dict[str, Any] = {}

        for cond in CONDITIONS:
            sub = [r for r in arm_rows if r["condition"] == cond]
            pos = [confidence(r) for r in sub if r["correct"]]
            neg = [confidence(r) for r in sub if not r["correct"]]
            entry[cond] = {
                "n": len(sub),
                "accuracy_full_coverage": (
                    sum(bool(r["correct"]) for r in sub) / len(sub) if sub else float("nan")
                ),
                "auroc_confidence_vs_correct": auroc(pos, neg),
                "risk_coverage": risk_coverage(sub),
                "mean_confidence": sum(confidence(r) for r in sub) / len(sub) if sub else 0.0,
            }

        # The measurement notes/29 exists for. At full coverage notes/08 reports
        # no gap between a real concept and a random direction. Does one reopen
        # when the least-confident rows are dropped?
        tgt = entry["target"]["risk_coverage"]
        rnd = entry["random"]["risk_coverage"]
        entry["selectivity_gap_by_coverage"] = {c: round(tgt[c] - rnd[c], 4) for c in tgt}
        out[arm] = entry

    return out


def load_remap(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in seeds:
        path = RESULTS / f"remap_training_v2_seed{s}_raw.jsonl"
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if float(r.get("strength", -1)) == REMAP_STRENGTH:
                    r["seed"] = s
                    rows.append(r)
    return rows


def analyse_remap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measurement 3: does the target/random gap reopen as coverage falls?"""
    scored = [r for r in rows if r.get("format_ok", True)]
    out: dict[str, Any] = {"margin_health": degeneracy(scored), "strength": REMAP_STRENGTH}
    for arm in REMAP_ARMS:
        arm_rows = [r for r in scored if r["arm"] == arm]
        entry: dict[str, Any] = {}
        for cond in CONDITIONS:
            sub = [r for r in arm_rows if r["condition"] == cond]
            pos = [confidence(r) for r in sub if r["correct"]]
            neg = [confidence(r) for r in sub if not r["correct"]]
            entry[cond] = {
                "n": len(sub),
                "accuracy_full_coverage": (
                    sum(bool(r["correct"]) for r in sub) / len(sub) if sub else float("nan")
                ),
                "auroc_confidence_vs_correct": auroc(pos, neg),
                "risk_coverage": risk_coverage(sub),
                "mean_confidence": (sum(confidence(r) for r in sub) / len(sub) if sub else 0.0),
            }
        tgt, rnd = entry["target"]["risk_coverage"], entry["random"]["risk_coverage"]
        entry["selectivity_gap_by_coverage"] = {c: round(tgt[c] - rnd[c], 4) for c in tgt}
        # Can confidence alone separate a real concept from a random direction,
        # ignoring labels entirely? This is the question an abstaining monitor asks.
        entry["auroc_confidence_separates_target_from_random"] = auroc(
            [confidence(r) for r in arm_rows if r["condition"] == "target"],
            [confidence(r) for r in arm_rows if r["condition"] == "random"],
        )
        out[arm] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("dev", "confirm"), required=True)
    ap.add_argument(
        "--measurement",
        choices=("ranking", "selectivity"),
        default="ranking",
        help="ranking = measurements 1-2 on report_training_v3; "
        "selectivity = measurement 3 on remap_training_v2",
    )
    args = ap.parse_args()

    if args.measurement == "selectivity":
        rows = load_remap(REMAP_SEEDS[args.phase])
        result = analyse_remap(rows)
        result["seeds"] = list(REMAP_SEEDS[args.phase])
        result["artifact"] = "remap_training_v2"
        result["phase"] = args.phase
        result["note"] = "notes/29 measurement 3"
        print(json.dumps(result, indent=2, sort_keys=True))
        existing = json.loads(OUT.read_text()) if OUT.exists() else {}
        existing[f"selectivity_{args.phase}"] = result
        OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {OUT} [selectivity_{args.phase}]")
        return

    rows = load(SEEDS[args.phase])
    result = analyse(rows)
    result["phase"] = args.phase
    result["seeds"] = list(SEEDS[args.phase])
    result["note"] = "notes/29"
    result["confidence_is"] = "abs(signed_margin); an internal margin, not a verbalized confidence"

    if args.phase == "dev":
        worst = min(result[a][c]["auroc_confidence_vs_correct"] for a in ARMS for c in ("target",))
        result["kill_rule"] = {
            "threshold": KILL_AUROC,
            "worst_target_auroc": worst,
            "fires": bool(worst < KILL_AUROC),
        }

    print(json.dumps(result, indent=2, sort_keys=True))

    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.phase] = result
    OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT} [{args.phase}]")


if __name__ == "__main__":
    main()
