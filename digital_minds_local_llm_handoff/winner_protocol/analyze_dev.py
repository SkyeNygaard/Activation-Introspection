#!/usr/bin/env python3
"""Analyze a bounded DEV cell and apply the Research-OS promotion gates.

DEV only: this script is for branch selection, not confirmatory inference.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty result file")
    return rows

def mean(xs):
    return float(np.mean(list(xs)))

def validate_grid(rows: list[dict]) -> dict:
    # One DEV runner file should be one model/layer/factor/persona cell.
    keys = ("model","vector_file","layer","factor","persona")
    for k in keys:
        vals = {json.dumps(r[k], sort_keys=True) for r in rows}
        if len(vals) != 1:
            raise ValueError(f"{k} varies inside one DEV artifact: {vals}")

    # Every carrier should have all 24 exact nuisance cells and balanced states/maps.
    by_carrier = defaultdict(list)
    for row in rows:
        by_carrier[row["carrier"]].append(row)
        if set(row["arms"]) != {"clean","query_only","target"}:
            raise ValueError("arm grid drifted")
        if row["query_sign"] not in (-1,1):
            raise ValueError("invalid query sign")
        if row["mapping_id"] not in (0,1):
            raise ValueError("invalid mapping")
        if row["correct_label"] not in ("Q","K"):
            raise ValueError("invalid label")

    for carrier, rs in by_carrier.items():
        if len(rs) != 24:
            raise ValueError(f"{carrier}: expected 24 cells, got {len(rs)}")
        if sum(r["query_sign"] == 1 for r in rs) != 12:
            raise ValueError(f"{carrier}: query signs unbalanced")
        if sum(r["mapping_id"] == 0 for r in rs) != 12:
            raise ValueError(f"{carrier}: mappings unbalanced")
        if sum(r["correct_label"] == "Q" for r in rs) != 12:
            raise ValueError(f"{carrier}: target labels unbalanced")
    return {"n_rows":len(rows), "n_carriers":len(by_carrier)}

def summarize(rows: list[dict], naive_rows=None, visible_rows=None) -> dict:
    grid = validate_grid(rows)
    carriers = sorted({r["carrier"] for r in rows})

    carrier_delta_p = {}
    carrier_target_acc = {}
    carrier_target_mass = {}
    carrier_target_format = {}
    carrier_semantic_state = {}

    for c in carriers:
        rs = [r for r in rows if r["carrier"] == c]
        carrier_delta_p[c] = mean(
            r["arms"]["target"]["p_correct"] - r["arms"]["query_only"]["p_correct"]
            for r in rs
        )
        carrier_target_acc[c] = mean(r["arms"]["target"]["correct"] for r in rs)
        carrier_target_mass[c] = mean(r["arms"]["target"]["label_mass"] for r in rs)
        carrier_target_format[c] = mean(r["arms"]["target"]["format_ok"] for r in rs)

        pos = [r["semantic_rating"] for r in rs if r["query_sign"] == +1]
        neg = [r["semantic_rating"] for r in rs if r["query_sign"] == -1]
        carrier_semantic_state[c] = mean(pos) - mean(neg)

    # Root gate 1: naive semantic intervention outside codebook context.
    naive_neutral_effect = None
    if naive_rows is not None:
        meta={(r["model"],int(r["layer"]),float(r["factor"])) for r in naive_rows}
        if meta != {(rows[0]["model"],int(rows[0]["layer"]),float(rows[0]["factor"]))}:
            raise ValueError("naive/model DEV metadata mismatch")
        by_carrier={}
        for c in sorted({r["carrier"] for r in naive_rows}):
            rs=[r for r in naive_rows if r["carrier"]==c and r["persona"]=="neutral"]
            if {r["sign"] for r in rs}!={-1,1}:
                raise ValueError("naive DEV lacks neutral +/- pair")
            plus=next(r["semantic_rating"] for r in rs if r["sign"]==1)
            minus=next(r["semantic_rating"] for r in rs if r["sign"]==-1)
            by_carrier[c]=plus-minus
        naive_neutral_effect=mean(by_carrier.values())

    # Root gate 2: visible-state arbitrary-code capability.
    visible_accuracy = None
    visible_format = None
    if visible_rows is not None:
        meta={(r["model"],int(r["layer"]),float(r["factor"])) for r in visible_rows}
        if meta != {(rows[0]["model"],int(rows[0]["layer"]),float(rows[0]["factor"]))}:
            raise ValueError("visible/model DEV metadata mismatch")
        if any(r["persona"]!="neutral" for r in visible_rows):
            raise ValueError("DEV visible control should be neutral persona")
        groups={}
        for c in sorted({r["carrier"] for r in visible_rows}):
            rs=[r for r in visible_rows if r["carrier"]==c]
            if len(rs)!=24:
                raise ValueError("visible DEV expects 24 cells/carrier")
            groups[c]=rs
        visible_accuracy=mean(
            mean(r["correct"] for r in rs) for rs in groups.values()
        )
        visible_format=mean(
            mean(r["format_ok"] for r in rs) for rs in groups.values()
        )

    result = {
        "grid": grid,
        "model": rows[0]["model"],
        "vector_file": rows[0]["vector_file"],
        "layer": rows[0]["layer"],
        "factor": rows[0]["factor"],
        "persona": rows[0]["persona"],
        "metrics": {
            "target_minus_query_only_p_correct": mean(carrier_delta_p.values()),
            "target_accuracy": mean(carrier_target_acc.values()),
            "target_label_mass": mean(carrier_target_mass.values()),
            "target_format_ok": mean(carrier_target_format.values()),
            "semantic_plus_minus_state_effect": mean(carrier_semantic_state.values()),
            "naive_semantic_plus_minus_neutral": naive_neutral_effect,
            "visible_control_accuracy": visible_accuracy,
            "visible_control_format": visible_format,
        },
        "carrier_values": {
            "target_minus_query_only_p_correct": carrier_delta_p,
            "semantic_plus_minus_state_effect": carrier_semantic_state,
        },
    }

    # DEV gates are intentionally descriptive / minimum-headroom gates.
    m = result["metrics"]
    semantic_gate_value = (
        m["naive_semantic_plus_minus_neutral"]
        if m["naive_semantic_plus_minus_neutral"] is not None
        else m["semantic_plus_minus_state_effect"]
    )
    gates = {
        "source_semantic_effect_positive": semantic_gate_value > 0.0,
        "visible_control_accuracy_at_least_0.90": (
            True if m["visible_control_accuracy"] is None
            else m["visible_control_accuracy"] >= 0.90
        ),
        "visible_control_format_at_least_0.90": (
            True if m["visible_control_format"] is None
            else m["visible_control_format"] >= 0.90
        ),
        "codebook_delta_at_least_0.10":
            m["target_minus_query_only_p_correct"] >= 0.10,
        "target_label_mass_at_least_0.90":
            m["target_label_mass"] >= 0.90,
        "target_format_at_least_0.90":
            m["target_format_ok"] >= 0.90,
    }
    result["dev_gates"] = {**gates, "passed": all(gates.values())}
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--naive-raw", type=Path)
    p.add_argument("--visible-raw", type=Path)
    a = p.parse_args()
    naive_rows = load_rows(a.naive_raw) if a.naive_raw else None
    visible_rows = load_rows(a.visible_raw) if a.visible_raw else None
    result = summarize(load_rows(a.raw), naive_rows, visible_rows)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")

if __name__ == "__main__":
    main()
