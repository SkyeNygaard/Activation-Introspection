"""Layer x strength sweep, all arms, with the observer comparison.

Writes one JSONL row per trial to results/ so analysis never re-runs the model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from introspect.analysis import headline, summarize_all
from introspect.concepts import DEFAULT_CONCEPTS, build_bank, max_offdiagonal_cosine
from introspect.experiment import TrialSet, run_cell
from introspect.models import DEFAULT_MODEL, load, memory_warning

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=str, default="", help="e.g. 8,14,20; default: 5 across depth")
    ap.add_argument("--strengths", type=str, default="0.05,0.1,0.2,0.4")
    ap.add_argument("--concepts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if (warn := memory_warning(args.model)) is not None:
        console.print(f"[yellow]warning: {warn}[/yellow]")

    model = load(args.model)
    console.print(
        f"{model.name}  layers={model.n_layers}  d_model={model.d_model}  "
        f"device={model.device.type}"
    )

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else [int(f * model.n_layers) for f in (0.25, 0.4, 0.55, 0.7, 0.85)]
    )
    strengths = [float(x) for x in args.strengths.split(",")]
    concepts = DEFAULT_CONCEPTS[: args.concepts]

    console.rule("concept bank")
    # Build once per layer -- the bank is layer-specific and rebuilding it inside
    # the sweep would be the dominant cost.
    banks = {}
    for layer in layers:
        bank = build_bank(model, layer, concepts=concepts)
        worst = max_offdiagonal_cosine(bank)
        banks[layer] = bank
        status = "[green]ok[/green]" if worst < 0.5 else "[red]DEGENERATE[/red]"
        console.print(f"  layer {layer:>3}  max|off-diag cosine| = {worst:.3f}  {status}")

    usable = [layer for layer in layers if max_offdiagonal_cosine(banks[layer]) < 0.5]
    if not usable:
        console.print(
            "[red]No layer produced a usable bank. Aborting: no result would mean anything.[/red]"
        )
        return

    console.rule("sweep")
    total = len(usable) * len(strengths) * len(concepts) * args.seeds
    console.print(f"{total} cells x 4 arms\n")

    ts = TrialSet()
    start = time.time()
    done = 0
    for layer in usable:
        for strength in strengths:
            for concept in concepts:
                for seed in range(args.seeds):
                    ts.trials.extend(
                        run_cell(model, banks[layer], concept, layer, strength, seed=seed)
                    )
                    done += 1
            rate = done / max(time.time() - start, 1e-9)
            console.print(
                f"  L{layer:<3} a={strength:<5} {done}/{total}  "
                f"{rate:.1f} cells/s  eta {(total - done) / max(rate, 1e-9) / 60:.1f} min"
            )

    out = args.out or Path("results") / f"sweep_{args.model.replace('/', '_')}.jsonl"
    ts.save(out)
    console.print(f"\nwrote {len(ts.trials)} trials to {out}")

    console.rule("summary")
    summaries = summarize_all(ts)
    table = Table("layer", "alpha", "KL", "det AUROC", "null AUROC", "identify", "observer", "gap")
    for s in summaries:
        table.add_row(
            str(s.layer),
            f"{s.strength}",
            f"{s.behavioural_kl.value:.4f}",
            f"{s.detection_auroc.value:.2f}",
            f"{s.detection_auroc_null.value:.2f}",
            f"{s.identify_acc.value:.2f}",
            f"{s.observer_acc.value:.2f}",
            f"{s.gap.value:+.2f} [{s.gap.lo:+.2f},{s.gap.hi:+.2f}]"
            + ("" if s.valid else " INVALID"),
        )
    console.print(table)
    console.print(f"\n[bold]{headline(summaries)}[/bold]")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
