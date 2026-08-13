#!/usr/bin/env python3
"""notes/33 -- three ways to boost introspection, one control.

Training, refusal ablation and prompting are each reported to make a model
introspect better. **All three audit false positives against trials where nothing
was injected.** The harder control -- norm-matched random directions, where
something real happened that meant nothing -- has existed since Lindsey's original
and base models pass it. notes/08 showed training fails it. notes/32 found ablation
is not a boost at this scale, so it cannot be audited. Prompting is the boost that
does transfer here (notes/24: anchor 0.694 -> 0.875) and nobody has audited it.

notes/14's design unchanged -- two concepts against two random directions at
identical class separation by construction -- crossed with three boost conditions.

Two things this adds over notes/32:

* a **prompting** arm, the third boost, never audited this way;
* **confidence margins**, so the abstention curve from notes/29 and notes/31 can be
  computed. ``score_pair`` in the frozen ``run_content_vs_disturbance`` returns
  correctness without a margin, so the scoring is re-implemented here identically
  apart from also returning the logit gap. The frozen function is not touched.

Inference only. No training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_content_vs_disturbance import PAIRS, matched_strength, pair_interventions, twin_pair
from run_heldout_elicitation import framed
from run_refusal_ablation_selectivity import ablate, build_direction
from run_zero_shot_identify import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import ConceptVector, build_bank, random_control
from introspect.hooks import intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

CONDITIONS = ("none", "prompt", "ablate")
ARMS = ("content", "random_pair")

#: notes/24's wording that measurably worked on this interface: it cut
#: constant-labelling from 40% to 25% and lifted the anchor to 0.875.
PROMPT_FAMILY = "introspect"

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)

#: notes/14 published this. The `none` arm must land within 0.10 of it.
NOTES_14_CONTENT_ACCURACY = 0.899
ANCHOR_TOLERANCE = 0.10


@torch.no_grad()
def score_with_margin(
    model: models.LoadedModel,
    prepared: Any,
    positive: ConceptVector,
    negative: ConceptVector,
    *,
    strength: float,
) -> dict[str, object]:
    """``score_pair``, plus the logit gap. Identical otherwise; frozen fn untouched."""
    interventions = pair_interventions(
        positive,
        negative,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=strength,
    )
    ids = prepared.input_ids
    with intervene(model, interventions, prompt_len=int(ids.shape[1])):
        logits = model.forward_logits(ids)[0, -1].float()
    label_ids = prepared.label_ids
    selected = logits[torch.tensor(label_ids, device=logits.device)]
    predicted = LABELS[int(selected.argmax())]
    top2 = torch.topk(selected, 2).values
    return {
        "predicted_label": predicted,
        "correct": predicted == prepared.episode.correct_label,
        "margin": float(top2[0] - top2[1]),
        "label_mass": float(torch.logsumexp(selected, 0).sub(torch.logsumexp(logits, 0)).exp()),
        "format_ok": int(logits.argmax()) in set(label_ids),
    }


def twin_pair_at_coverage(rows: list[dict[str, Any]], coverage: float) -> float:
    """Twin-pair accuracy over the most-confident cells. notes/29's instrument.

    A twin pair's confidence is the smaller of its two members' margins: a cell is
    only as trustworthy as its weaker half.
    """
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["pair"], r["carrier_sha"], r["cell_base"]), []).append(r)
    full = [v for v in by.values() if len(v) == 2]
    if not full:
        return float("nan")
    ranked = sorted(full, key=lambda v: min(x["margin"] for x in v), reverse=True)
    k = max(1, round(len(ranked) * coverage))
    kept = ranked[:k]
    return sum(all(x["correct"] for x in v) for v in kept) / len(kept)


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

        print("building refusal direction (carried control)", flush=True)
        _layer, direction, gate = build_direction(model)
        print(f"  gate passed: {gate['gate_passed']}", flush=True)

        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([c.vector for c in centering.values()]).mean(0)
        names = sorted({n for p in PAIRS for n in p})
        raw = build_bank(model, LAYER, names, center=False)
        bank = {
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in raw.items()
        }

        carrier = CONFIRM_VISIBLE_SAMPLES[0]
        carrier_sha = hashlib.sha256(carrier.encode()).hexdigest()[:16]
        base_episodes = exact_episodes(carrier)
        if args.smoke:
            base_episodes = base_episodes[:2]

        # `prompt` differs only in the instruction header; every line carrying an
        # injection site is byte-identical, so the input-only control still holds.
        prepared = {
            "plain": [prepare_episode(model, e) for e in base_episodes],
            "framed": [prepare_episode(model, framed(e, PROMPT_FAMILY)) for e in base_episodes],
        }

        rows: list[dict[str, object]] = []
        pairs = PAIRS[:1] if args.smoke else PAIRS
        for cond in CONDITIONS:
            preps = prepared["framed"] if cond == "prompt" else prepared["plain"]
            for a_name, b_name in pairs:
                a, b = bank[a_name], bank[b_name]
                for arm in ARMS:
                    if arm == "content":
                        pos, neg = a, b
                    else:
                        pos = random_control(a, seed=hash(a_name) % 10000)
                        neg = random_control(b, seed=hash(b_name) % 10000)
                    strength = matched_strength(pos.vector, neg.vector)
                    for prep in preps:
                        if cond == "ablate":
                            with ablate(model, direction):
                                r = score_with_margin(model, prep, pos, neg, strength=strength)
                        else:
                            r = score_with_margin(model, prep, pos, neg, strength=strength)
                        rows.append(
                            {
                                "condition": cond,
                                "arm": arm,
                                # condition must be inside the twin key or cells from
                                # different conditions collide.
                                "pair": f"{a_name}|{b_name}|{cond}",
                                "carrier_sha": carrier_sha,
                                "cell_base": prep.episode.cell_id.rsplit("q", 1)[0],
                                "cell_id": prep.episode.cell_id,
                                "strength": strength,
                                **r,
                            }
                        )
            print(f"  {cond} done ({time.time() - started:.0f}s)", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        table: dict[str, Any] = {}
        for cond in CONDITIONS:
            entry: dict[str, Any] = {}
            for arm in ARMS:
                sub = [r for r in rows if r["condition"] == cond and r["arm"] == arm]
                entry[arm] = {
                    "n": len(sub),
                    "accuracy": sum(bool(r["correct"]) for r in sub) / len(sub),
                    "twin_pair": twin_pair(sub),
                    "format_rate": sum(bool(r["format_ok"]) for r in sub) / len(sub),
                    "mean_margin": sum(float(r["margin"]) for r in sub) / len(sub),
                }
            # The abstention curve: does the gap reopen or close as the least
            # confident cells are dropped? notes/29 and notes/31 turn on this.
            gaps = {}
            for cov in COVERAGES:
                c = twin_pair_at_coverage(
                    [r for r in rows if r["condition"] == cond and r["arm"] == "content"], cov
                )
                q = twin_pair_at_coverage(
                    [r for r in rows if r["condition"] == cond and r["arm"] == "random_pair"],
                    cov,
                )
                gaps[f"{cov:.1f}"] = round(c - q, 4)
            entry["selectivity_gap_by_coverage"] = gaps
            table[cond] = entry

        anchor = table["none"]["content"]["accuracy"]
        anchor_ok = abs(anchor - NOTES_14_CONTENT_ACCURACY) <= ANCHOR_TOLERANCE
        summary = {
            "note": "notes/33",
            "model": MODEL,
            "injection_layer": LAYER,
            "prompt_family": PROMPT_FAMILY,
            "refusal_gate": gate,
            "anchor": {
                "none_content_accuracy": anchor,
                "notes_14_published": NOTES_14_CONTENT_ACCURACY,
                "within_tolerance": anchor_ok,
            },
            "table": table,
            "smoke": bool(args.smoke),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary["table"], indent=2, sort_keys=True), flush=True)
        if not anchor_ok and not args.smoke:
            print(
                "\nKILL RULE: the `none` arm does not reproduce notes/14. "
                "The apparatus has drifted; read no boost arm.",
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
