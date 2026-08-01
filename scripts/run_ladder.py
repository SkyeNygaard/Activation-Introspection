"""Run the same design across the scale ladder, one model at a time.

The scale question is the point: a null result at 0.5B says little, and the
interesting claim is whether the introspector-observer gap *opens up* with size.
Models are loaded and freed one at a time -- holding two resident is what pushes
a 24 GB machine into swap, where a sweep silently takes hours.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from introspect.analysis import summarize_all
from introspect.concepts import DEFAULT_CONCEPTS, build_bank, max_offdiagonal_cosine
from introspect.experiment import TrialSet, run_cell
from introspect.models import SCALE_LADDER, load, memory_warning

console = Console()


def sweep_model(
    name: str, layers_frac: list[float], strengths: list[float], concepts: list[str], seeds: int
) -> TrialSet:
    if (warn := memory_warning(name)) is not None:
        console.print(f"[yellow]{warn}[/yellow]")

    model = load(name)
    layers = sorted({int(f * model.n_layers) for f in layers_frac})
    console.print(f"[bold]{name}[/bold]  {model.n_layers} layers, probing {layers}")

    ts = TrialSet()
    for layer in layers:
        bank = build_bank(model, layer, concepts=concepts)
        worst = max_offdiagonal_cosine(bank)
        if worst >= 0.5:
            console.print(f"  layer {layer}: bank degenerate ({worst:.2f}), skipping")
            continue
        for strength in strengths:
            t0 = time.time()
            for concept in concepts:
                for seed in range(seeds):
                    ts.trials.extend(run_cell(model, bank, concept, layer, strength, seed=seed))
            console.print(
                f"  L{layer:<3} a={strength:<5} cos={worst:.2f}  "
                f"{len(concepts) * seeds} cells in {time.time() - t0:.0f}s"
            )

    model.free()
    gc.collect()
    return ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(SCALE_LADDER))
    ap.add_argument("--layer-fracs", default="0.4,0.6,0.8")
    ap.add_argument("--strengths", default="0.05,0.1,0.2")
    ap.add_argument("--concepts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/ladder.jsonl"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.layer_fracs.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    concepts = DEFAULT_CONCEPTS[: args.concepts]

    combined = TrialSet()
    for name in args.models.split(","):
        combined.trials.extend(sweep_model(name, fracs, strengths, concepts, args.seeds).trials)
        combined.save(args.out)  # checkpoint after each model

    console.rule("scale ladder")
    table = Table(
        "model", "layer", "alpha", "KL", "det AUROC", "null", "identify", "observer", "gap"
    )
    for s in summarize_all(combined):
        table.add_row(
            s.model.split("/")[-1],
            str(s.layer),
            f"{s.strength}",
            f"{s.behavioural_kl.value:.3f}",
            f"{s.detection_auroc.value:.2f}",
            f"{s.detection_auroc_null.value:.2f}",
            f"{s.identify_acc.value:.2f}",
            f"{s.observer_acc.value:.2f}",
            f"{s.gap.value:+.2f} [{s.gap.lo:+.2f},{s.gap.hi:+.2f}]"
            + ("" if s.valid else " INVALID"),
        )
    console.print(table)
    console.print(f"\nwrote {len(combined.trials)} trials to {args.out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
