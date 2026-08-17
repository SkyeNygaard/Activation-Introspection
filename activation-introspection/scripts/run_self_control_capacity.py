"""Capacity check for notes/40: told plainly to think about X, does the state move toward X?

Pre-run note: ``notes/40-can-it-move-its-own-state.md``, written before this ran.

This is the cheapest possible version of the question, with the answer in plain
sight: no code, no demonstrations, no hidden mapping. The instruction names the
concept outright. If the state does not move preferentially toward the named
concept's direction here, no design downstream can work and the line closes.

A pass is a licence to build, not a finding. The instruction contains the concept
word, so a state sitting on that concept's direction may reflect the word being in
the context rather than any steering. Only the twin design in notes/40 separates
those, and this script does not attempt to.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_content_vs_disturbance import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.concepts import build_bank
from introspect.hooks import capture
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

#: The bank the rest of the repository uses, so a pass transfers to the twin design.
from introspect.concepts import DEFAULT_CONCEPTS

#: Several wordings, because a single instruction confounds the ability with one
#: phrasing. Declared before the run; no wording is dropped after seeing results.
INSTRUCTIONS = (
    "Think about {concept}.",
    "Bring {concept} to mind and hold it there.",
    "Focus your attention on {concept}.",
    "Concentrate on the idea of {concept}.",
)

#: Names no concept. Establishes where the projections sit with nothing asked for.
NEUTRAL = (
    "Think about nothing in particular.",
    "Let your attention rest wherever it happens to be.",
    "Hold your mind still.",
    "Wait quietly.",
)


def state_for(model: models.LoadedModel, text: str, layer: int) -> torch.Tensor:
    """Residual stream at the last token of the rendered chat prompt."""
    ids = model.encode(model.chat(text))
    with capture(model, [layer]) as store:
        model.forward_logits(ids)
    return store.acts[layer][0][0, -1].float().cpu()


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

        # Built exactly as the rest of the repository builds it: the centering
        # offset comes from a separate concept set, not from these eight.
        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
        raw = build_bank(model, LAYER, list(DEFAULT_CONCEPTS), center=False)
        concepts = sorted(raw)
        dirs = torch.stack(
            [(raw[c].vector - center) / (raw[c].vector - center).norm() for c in concepts]
        ).float()
        print(f"bank built: {concepts}", flush=True)

        rows: list[dict[str, object]] = []
        for concept in concepts:
            for wording, template in enumerate(INSTRUCTIONS):
                state = state_for(model, template.format(concept=concept), LAYER)
                proj = (dirs @ state).tolist()
                order = sorted(range(len(concepts)), key=lambda i: -proj[i])
                rank = order.index(concepts.index(concept))
                rows.append(
                    {
                        "kind": "instructed",
                        "concept": concept,
                        "wording": wording,
                        "rank_of_named": rank,
                        "top1": rank == 0,
                        "projections": dict(zip(concepts, proj, strict=True)),
                    }
                )
            print(f"  {concept}: top1 {sum(bool(r['top1']) for r in rows if r['concept']==concept)}/4", flush=True)

        for wording, text in enumerate(NEUTRAL):
            state = state_for(model, text, LAYER)
            proj = (dirs @ state).tolist()
            rows.append(
                {
                    "kind": "neutral",
                    "concept": None,
                    "wording": wording,
                    "rank_of_named": None,
                    "top1": None,
                    "projections": dict(zip(concepts, proj, strict=True)),
                }
            )

        instructed = [r for r in rows if r["kind"] == "instructed"]
        top1 = sum(bool(r["top1"]) for r in instructed) / len(instructed)
        mean_rank = sum(int(r["rank_of_named"]) for r in instructed) / len(instructed)  # type: ignore[arg-type]
        by_wording = {
            w: sum(bool(r["top1"]) for r in instructed if r["wording"] == w)
            / len([r for r in instructed if r["wording"] == w])
            for w in range(len(INSTRUCTIONS))
        }
        summary = {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "n_concepts": len(concepts),
            "chance_top1": 1.0 / len(concepts),
            "chance_mean_rank": (len(concepts) - 1) / 2,
            "top1_rate": top1,
            "mean_rank_of_named": mean_rank,
            "top1_by_wording": by_wording,
            "n_instructed": len(instructed),
            "elapsed_s": round(time.time() - started, 1),
            "reading": (
                "top1 well above 1/8 -> the model steers toward a named target; build the "
                "twin design. Moves but not toward the named concept -> not target-directed. "
                "At chance -> notes/40 kill rule fires, close the line."
            ),
            "caveat": (
                "The instruction contains the concept word, so a pass may reflect the word "
                "being in context rather than steering. Only the twin design separates them."
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/self_control_capacity_v1.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
