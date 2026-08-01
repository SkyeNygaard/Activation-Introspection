"""Present-but-unreportable, or absent? Probe vs self-report at the answer position."""

from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.table import Table

from introspect.concepts import DEFAULT_CONCEPTS, build_bank, max_offdiagonal_cosine
from introspect.models import DEFAULT_MODEL, load
from introspect.probe import run_probe

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--inject-layer", type=int, default=None)
    ap.add_argument("--strength", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    model = load(args.model)
    inject = args.inject_layer if args.inject_layer is not None else int(0.4 * model.n_layers)
    bank = build_bank(model, inject, concepts=DEFAULT_CONCEPTS)
    worst = max_offdiagonal_cosine(bank)
    console.print(f"{model.name}  inject at L{inject}  bank max|off-diag|={worst:.2f}")
    if worst >= 0.5:
        console.print("[red]Degenerate bank; aborting.[/red]")
        return

    # Probe strictly downstream of the injection, so the question is what
    # survives into the answer state rather than whether the vector is there.
    probe_layers = sorted({inject + 2, int(0.7 * model.n_layers), model.n_layers - 1})
    probe_layers = [layer for layer in probe_layers if inject < layer < model.n_layers]

    table = Table(
        "probe layer",
        "n",
        "raw probe (circular)",
        "ABLATED probe",
        "shuffled null",
        "self-report",
        "gap",
    )
    results = []
    for probe_layer in probe_layers:
        r = run_probe(model, bank, inject, probe_layer, args.strength, seeds=range(args.seeds))
        results.append(r)
        table.add_row(
            str(probe_layer),
            str(r.n),
            f"{r.probe_acc_raw.value:.2f}",
            str(r.probe_acc_ablated),
            str(r.probe_acc_shuffled),
            str(r.self_report_acc),
            str(r.gap),
        )
    console.print(table)
    console.print(f"\n[dim]chance = {1 / len(bank):.3f}[/dim]")
    best = max(results, key=lambda r: r.gap.value)
    console.print(f"\n[bold]{best.verdict}[/bold]")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
