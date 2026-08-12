#!/usr/bin/env python3
"""Rescore a matched-reader run at the twin-pair unit. Evidence for notes/22.

notes/15 scored the weak arm by row and read 0.497 as chance. The frozen
protocol's inference unit is the twin pair, where the same rows give 0.007 --
a constant-label floor, not chance. This recomputes that table from saved raw
rows so the claim does not depend on shell history.
"""

import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "results" / "matched_reader_content_v1_raw.jsonl"
READER = "reader_centroid_euclidean_correct"


def rescore(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group byte-identical query twins and score each pair jointly."""
    out: dict[str, dict[str, Any]] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        arm_rows = [r for r in rows if r["arm"] == arm]
        twins: dict[tuple[str, str, str], dict[int, Any]] = collections.defaultdict(dict)
        for r in arm_rows:
            # cell_id is o<order>m<mapping>q<sign>; the two signs are one pair
            base = r["cell_id"].split("q")[0]
            twins[(base, r["carrier_sha"], r["pair"])][r["query_sign"]] = r
        pairs = [v for v in twins.values() if len(v) == 2]
        model = [all(x["model_correct"] for x in v.values()) for v in pairs]
        reader = [all(x[READER] for x in v.values()) for v in pairs]

        # The reader is fit per cell, so its verdict should be constant within
        # one. If that holds, the 288 rows are 24 independent units, not 288.
        by_cell: dict[str, set[bool]] = collections.defaultdict(set)
        for r in arm_rows:
            by_cell[r["cell_id"]].add(r[READER])

        out[arm] = {
            "n_rows": len(arm_rows),
            "n_pairs": len(pairs),
            "row_model": sum(r["model_correct"] for r in arm_rows) / len(arm_rows),
            "row_reader": sum(r[READER] for r in arm_rows) / len(arm_rows),
            "pair_model": sum(model) / len(pairs),
            "pair_reader": sum(reader) / len(pairs),
            "pair_model_only": sum(m and not d for m, d in zip(model, reader, strict=True)),
            "n_cells": len(by_cell),
            "cells_where_reader_verdict_varies": sum(1 for v in by_cell.values() if len(v) > 1),
        }
    return out


def constant_label_cells(rows: list[dict[str, Any]], arm: str) -> list[tuple[str, str, int]]:
    """Cells where the model emits one label regardless of the injected sign."""
    arm_rows = [r for r in rows if r["arm"] == arm]
    by_cell: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in arm_rows:
        by_cell[r["cell_id"]].append(r)
    return [
        (cell, next(iter({r["model_predicted"] for r in v})), len(v))
        for cell, v in sorted(by_cell.items())
        if len({r["model_predicted"] for r in v}) == 1
    ]


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    table = rescore(rows)

    print(f"{raw.name}: {len(rows)} rows\n")
    print(
        f"{'arm':16s} {'row:mdl':>8s} {'row:rdr':>8s} "
        f"{'pair:mdl':>9s} {'pair:rdr':>9s} {'mdl-only':>9s}"
    )
    for arm, s in table.items():
        print(
            f"{arm:16s} {s['row_model']:8.3f} {s['row_reader']:8.3f} "
            f"{s['pair_model']:9.3f} {s['pair_reader']:9.3f} "
            f"{s['pair_model_only']:9d}"
        )

    weak = next((a for a in table if "weak" in a), None)
    if weak:
        const = constant_label_cells(rows, weak)
        print(
            f"\n{weak}: {len(const)} of "
            f"{len({r['cell_id'] for r in rows if r['arm'] == weak})} cells emit "
            f"a constant label across both query signs"
        )
        for cell, label, n in const:
            print(f"  {cell:12s} {label}x{n}")

    # notes/22's two claims, as assertions
    if weak and raw == DEFAULT_RAW:
        s = table[weak]
        assert s["pair_model"] < 0.05, f"weak-arm pair accuracy {s['pair_model']}"
        assert s["row_model"] > 0.45, "row accuracy no longer reads as chance"
        assert s["cells_where_reader_verdict_varies"] == 0, "reader varies in-cell"
        print("\nok: weak arm is a floor at the pair unit while rows read as chance")


if __name__ == "__main__":
    main()
