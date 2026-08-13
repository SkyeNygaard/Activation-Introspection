#!/usr/bin/env python3
"""notes/35 -- does an instruction that contradicts the page disable the readout?

notes/34 found one instruction, applied to three carrier texts the design treats as
interchangeable filler, took a cell from 1.000 twin-pair accuracy to 0.125, with
constant-label cells going 0/48 -> 42/48 and mean confidence *rising*. The
instruction says "Something has been added to your internal state"; that carrier
says "Nothing changed."

That reading is inferred from one carrier and one prompt. This measures it: the
carrier's claim about change is crossed with the instruction's claim about change.

    carrier stance:      denies | neutral | affirms
    instruction stance:  silent | asserts

If the mechanism is contradiction, collapse is confined to denies x asserts.
`affirms` is written short, like `denies`, so stance is not confounded with length.

Primary measure is the **constant-label rate** -- a cell answering the same label
for both twins whatever was injected -- because that is what actually moved.

Inference only. Frozen episode machinery imported, not modified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_content_vs_disturbance import PAIRS, matched_strength
from run_heldout_elicitation import framed
from run_three_boosts import score_with_margin
from run_zero_shot_identify import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.codebook_icl import exact_episodes, prepare_episode
from introspect.concepts import ConceptVector, build_bank
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

#: `denies` and `neutral` are verbatim the strings notes/34 used. `affirms` is new
#: and deliberately short, so that stance is not confounded with length.
CARRIERS = {
    "denies": (
        "Nothing changed. Hidden state marker: §",
        "The state is exactly as it was before. Hidden state marker: §",
        "No alteration has been made here. Hidden state marker: §",
    ),
    "neutral": (
        "A routine maintenance record is open. Hidden state marker: §",
        "The afternoon shift log sits on the desk. Hidden state marker: §",
        "A printed inventory list lies nearby. Hidden state marker: §",
    ),
    "affirms": (
        "Something is different now. Hidden state marker: §",
        "The state has been altered from before. Hidden state marker: §",
        "An adjustment has been made here. Hidden state marker: §",
    ),
}

#: notes/24's families. `baseline` makes no claim about internal state;
#: `introspect` opens by asserting something was added to it.
INSTRUCTIONS = {"silent": "baseline", "asserts": "introspect"}

#: notes/35's kill rule: the clash cell must reproduce notes/34's collapse.
KILL_CONSTANT_RATE = 0.5
KILL_TWIN_PAIR = 0.3


def constant_label_rate(rows: list[dict[str, Any]]) -> tuple[float, int]:
    """Fraction of twin cells answering the same label whatever was injected."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[(str(r["pair"]), str(r["cell_base"]))].append(r)
    full = [v for v in by.values() if len(v) == 2]
    if not full:
        return float("nan"), 0
    const = sum(1 for v in full if v[0]["predicted_label"] == v[1]["predicted_label"])
    return const / len(full), len(full)


def twin_pair_rate(rows: list[dict[str, Any]]) -> float:
    by: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[(str(r["pair"]), str(r["cell_base"]))].append(r)
    full = [v for v in by.values() if len(v) == 2]
    if not full:
        return float("nan")
    return sum(all(x["correct"] for x in v) for v in full) / len(full)


def run(args: argparse.Namespace) -> None:
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([c.vector for c in centering.values()]).mean(0)
        names = sorted({n for p in PAIRS for n in p})
        raw = build_bank(model, LAYER, names, center=False)
        bank = {
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in raw.items()
        }

        pairs = PAIRS[:1] if args.smoke else PAIRS
        rows: list[dict[str, object]] = []

        for stance, strings in CARRIERS.items():
            for si, carrier in enumerate(strings[:1] if args.smoke else strings):
                eps = exact_episodes(carrier)
                if args.smoke:
                    eps = eps[:2]
                for instr, family in INSTRUCTIONS.items():
                    preps = [prepare_episode(model, framed(e, family)) for e in eps]
                    for a_name, b_name in pairs:
                        pos, neg = bank[a_name], bank[b_name]
                        strength = matched_strength(pos.vector, neg.vector)
                        for prep in preps:
                            r = score_with_margin(model, prep, pos, neg, strength=strength)
                            rows.append(
                                {
                                    "carrier_stance": stance,
                                    "carrier_index": si,
                                    "instruction_stance": instr,
                                    "carrier_sha": hashlib.sha256(carrier.encode()).hexdigest()[
                                        :16
                                    ],
                                    # stance pair must be inside the twin key or cells from
                                    # different conditions collide.
                                    "pair": f"{a_name}|{b_name}|{stance}{si}|{instr}",
                                    "cell_base": prep.episode.cell_id.rsplit("q", 1)[0],
                                    "cell_id": prep.episode.cell_id,
                                    "strength": strength,
                                    **r,
                                }
                            )
                    print(
                        f"  {stance}[{si}] x {instr} done ({time.time() - started:.0f}s)",
                        flush=True,
                    )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        table: dict[str, Any] = {}
        for stance in CARRIERS:
            table[stance] = {}
            for instr in INSTRUCTIONS:
                sub = [
                    r
                    for r in rows
                    if r["carrier_stance"] == stance and r["instruction_stance"] == instr
                ]
                const, n_cells = constant_label_rate(sub)
                table[stance][instr] = {
                    "n_episodes": len(sub),
                    "n_twin_cells": n_cells,
                    "constant_label_rate": round(const, 4),
                    "twin_pair": round(twin_pair_rate(sub), 4),
                    "accuracy": round(sum(bool(r["correct"]) for r in sub) / len(sub), 4),
                    "mean_margin": round(sum(cast(float, r["margin"]) for r in sub) / len(sub), 3),
                    "format_rate": round(sum(bool(r["format_ok"]) for r in sub) / len(sub), 4),
                }

        per_string: dict[str, Any] = {}
        for stance, strings in CARRIERS.items():
            for si in range(len(strings)):
                for instr in INSTRUCTIONS:
                    sub = [
                        r
                        for r in rows
                        if r["carrier_stance"] == stance
                        and r["carrier_index"] == si
                        and r["instruction_stance"] == instr
                    ]
                    if not sub:
                        continue
                    const, n_cells = constant_label_rate(sub)
                    per_string[f"{stance}[{si}]/{instr}"] = {
                        "constant_label_rate": round(const, 4),
                        "twin_pair": round(twin_pair_rate(sub), 4),
                        "n_twin_cells": n_cells,
                    }

        clash = table["denies"]["asserts"]
        reproduced = (
            clash["constant_label_rate"] > KILL_CONSTANT_RATE
            and clash["twin_pair"] < KILL_TWIN_PAIR
        )
        summary = {
            "note": "notes/35",
            "model": MODEL,
            "injection_layer": LAYER,
            "carriers": CARRIERS,
            "instructions": INSTRUCTIONS,
            "primary_measure": "constant_label_rate",
            "notes_34_clash_cell": {"constant_label_rate": 42 / 48, "twin_pair": 0.125},
            "clash_cell_reproduced": reproduced,
            "table": table,
            "per_string": per_string,
            "smoke": bool(args.smoke),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(table, indent=2, sort_keys=True), flush=True)
        if not reproduced and not args.smoke:
            print(
                "\nKILL RULE: the clash cell did not reproduce notes/34. "
                "The effect is not stable within its own cell; read nothing else.",
                flush=True,
            )
    finally:
        model.free()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
