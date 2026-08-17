"""notes/44 -- can a fitted probe follow the model to the answer position?

Pre-run note: ``notes/44-can-a-fitted-probe-follow-the-model.md``, written before this ran.

notes/18 found the logit lens at chance at the answer position for nineteen
consecutive blocks while the model identified the concept at 0.667, and called it
"the first result here that looks like the model doing work a cheap reader cannot
follow". It was never a claim because the lens is an *unfitted* readout. This fits a
supervised probe on the same states and asks whether it can follow.

The probe is deliberately given every advantage: supervised labels the model never
sees, and a random cross-validation split rather than a by-carrier one. A shuffled
label null runs beside every cell, because in 2048 dimensions a probe separates
almost anything -- the artifact notes/38 caught.

Nothing is reimplemented: the sweep and the lens come from ``run_lens_depth``, the
probe from ``introspect.probe``, the carriers from notes/41's blind set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_lens_depth import lens_all
from run_zero_shot_identify import (
    ANSWER_PREFIX,
    CHANCE,
    LAYER,
    MARKER,
    MODEL,
    MODEL_REVISION,
    first_token_ids,
    question,
)

from introspect import models
from introspect.codebook_icl import CONFIRM_CONCEPTS, CONFIRM_VISIBLE_SAMPLES
from introspect.concepts import ConceptVector, build_bank
from introspect.hooks import Intervention, capture, intervene
from introspect.preflight import check as preflight_check
from introspect.probe import fit_probe
from introspect.report_training import CENTERING_CONCEPTS

STRENGTH = 2.0
SITES = ("marker", "final")

#: notes/18 measured 1.000 here. If this moves, the harness moved.
ANCHOR_SITE, ANCHOR_DEPTH, ANCHOR_EXPECTED = "marker", 9, 1.000
ANCHOR_TOLERANCE = 0.10

BLIND_CARRIERS = Path("results/blind_carriers_v1.json")
N_BLIND = 8


def carriers_for(smoke: bool) -> list[str]:
    if smoke:
        # Two, not one: the smoke must exercise the probe fit, which needs
        # at least one example per class per fold.
        return list(CONFIRM_VISIBLE_SAMPLES[:2])
    blind = json.loads(BLIND_CARRIERS.read_text())["sentences"][:N_BLIND]
    return list(CONFIRM_VISIBLE_SAMPLES) + [f"{s.rstrip()} Hidden state marker: {MARKER}" for s in blind]


def run(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {out}")

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)
        depths = list(range(LAYER, len(model.blocks)))

        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
        rawbank = build_bank(model, LAYER, list(CONFIRM_CONCEPTS), center=False)
        bank = {
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in rawbank.items()
        }
        options = list(CONFIRM_CONCEPTS)
        concepts = options[:2] if args.smoke else options
        carriers = carriers_for(args.smoke)
        print(f"{len(carriers)} carriers x {len(concepts)} concepts, depths {depths[0]}..{depths[-1]}",
              flush=True)

        # states[(depth, site)] -> list of vectors, aligned with `labels`
        states: dict[tuple[int, str], list[np.ndarray]] = {}
        lens_ok: dict[tuple[int, str], list[bool]] = {}
        labels: list[str] = []
        model_correct: list[bool] = []
        carrier_of: list[str] = []

        for carrier in carriers:
            prompt = model.chat(f"{carrier}\n\n" + question(options), ANSWER_PREFIX)
            ids = model.encode(prompt)
            option_ids = first_token_ids(model, prompt, options)
            marker_pos = None
            for pos, tok in enumerate(ids[0].tolist()):
                if MARKER in model.tokenizer.decode([tok]):
                    marker_pos = pos
            if marker_pos is None:
                raise ValueError("marker token not found")
            final_pos = int(ids.shape[1]) - 1

            for concept in concepts:
                edit = Intervention(
                    layer=LAYER, direction=bank[concept].vector, strength=STRENGTH,
                    positions=[marker_pos], per_position=True, label=concept,
                )
                with intervene(model, [edit], prompt_len=int(ids.shape[1])), \
                        capture(model, depths) as store:
                    logits = model.forward_logits(ids)[0, -1].float().cpu()

                # The model's own answer on this very episode.
                model_correct.append(options[int(logits[option_ids].argmax())] == concept)
                labels.append(concept)
                carrier_of.append(carrier[:24])

                for depth in depths:
                    acts = store.acts[depth][0][0]
                    for site, pos in (("marker", marker_pos), ("final", final_pos)):
                        vec = acts[pos].float().cpu()
                        states.setdefault((depth, site), []).append(vec.numpy())
                        pick = options[int(lens_all(model, vec.clone())[option_ids].argmax())]
                        lens_ok.setdefault((depth, site), []).append(pick == concept)
            print(f"  {carrier[:26]}  ({time.time()-started:.0f}s)", flush=True)

        y = np.array(labels)
        n = len(y)
        # StratifiedKFold needs at least one sample per class per fold. The real run
        # has 11 per class; a smoke has 1, so cap the folds by the rarest class
        # rather than crashing on a configuration that is only ever used for plumbing.
        per_class = min(int((y == c).sum()) for c in set(labels))
        n_splits = max(2, min(5, per_class))
        print(f"captured {n} episodes, {per_class} per concept; {n_splits}-fold probes",
              flush=True)

        cells: dict[str, Any] = {}
        for depth in depths:
            for site in SITES:
                x = np.stack(states[(depth, site)])
                real = fit_probe(x, y, seed=0, n_splits=n_splits)
                null = fit_probe(x, y, seed=0, n_splits=n_splits, shuffle_labels=True)
                cells[f"{site}@{depth}"] = {
                    "probe": sum(real) / n,
                    "probe_shuffled_null": sum(null) / n,
                    "lens": sum(lens_ok[(depth, site)]) / n,
                }

        model_acc = sum(model_correct) / n
        anchor = cells[f"{ANCHOR_SITE}@{ANCHOR_DEPTH}"]["lens"]
        summary = {
            "model": MODEL, "inject_layer": LAYER, "strength": STRENGTH,
            "n_episodes": n, "n_carriers": len(carriers), "chance": CHANCE,
            "probe_folds": n_splits, "episodes_per_concept": per_class,
            "model_forced_choice": model_acc,
            "anchor": {
                "cell": f"lens {ANCHOR_SITE}@{ANCHOR_DEPTH}", "measured": anchor,
                "notes_18_expected": ANCHOR_EXPECTED,
                "within_tolerance": abs(anchor - ANCHOR_EXPECTED) <= ANCHOR_TOLERANCE,
            },
            "cells": cells,
            "elapsed_s": round(time.time() - started, 1),
            "reading": (
                "probe at chance at site 'final' where the model is well above it -> "
                "information the model uses that a fitted readout cannot recover there. "
                "probe recovers it -> notes/18 was an unfitted-readout artifact and closes."
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))

        print(f"\nmodel forced choice on these {n} episodes: {model_acc:.3f}  (chance {CHANCE})")
        print(f"anchor lens marker@{ANCHOR_DEPTH}: {anchor:.3f} vs notes/18 {ANCHOR_EXPECTED}")
        print(f"\n{'depth':>5} {'lens_mk':>8} {'probe_mk':>9} {'lens_fin':>9} {'probe_fin':>10} {'null_fin':>9}")
        for depth in depths:
            m, f = cells[f"marker@{depth}"], cells[f"final@{depth}"]
            print(f"{depth:5d} {m['lens']:8.3f} {m['probe']:9.3f} {f['lens']:9.3f} "
                  f"{f['probe']:10.3f} {f['probe_shuffled_null']:9.3f}")
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/probe_depth_v1.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
