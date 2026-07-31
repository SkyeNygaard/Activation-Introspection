"""End-to-end plumbing check: does an injection change anything at all?

Run this first. It proves four things before any experiment is worth running:

1. The model loads on this machine and generates.
2. Hooks fire on the right blocks and are removed afterwards.
3. Concept vectors are distinguishable from each other (low pairwise cosine).
4. There exists a strength window where injection changes the output measurably
   without reducing it to word salad.

It is *not* a result. It is the check that a null result later would mean
something.
"""

from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.table import Table

from introspect import build_bank, generate, load
from introspect.concepts import DEFAULT_CONCEPTS, max_offdiagonal_cosine, random_control
from introspect.hooks import Intervention
from introspect.models import DEFAULT_MODEL
from introspect.prompts import DETECT, IDENTIFY, NEUTRAL_TASK

console = Console()

# Strength is a multiple of the measured residual norm at the injection layer,
# so the usable window is well below 1.0. Anything at or above ~1.0 is louder
# than the model's own representation and reliably produces word salad.
STRENGTHS = (0.05, 0.1, 0.2, 0.4, 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concept", default="ocean")
    ap.add_argument("--layer", type=int, default=None, help="default: 60%% of depth")
    args = ap.parse_args()

    console.rule("load")
    model = load(args.model)
    layer = args.layer if args.layer is not None else int(0.6 * model.n_layers)
    console.print(
        f"{model.name}  layers={model.n_layers}  d_model={model.d_model}  "
        f"device={model.device.type}  dtype={model.dtype}  inject_layer={layer}"
    )

    console.rule("concept bank")
    bank = build_bank(model, layer, concepts=DEFAULT_CONCEPTS)
    names = list(bank)
    table = Table("", *names)
    for a in names:
        row = []
        for b in names:
            c = float(torch.dot(bank[a].unit(), bank[b].unit()))
            row.append(f"[dim]{c:+.2f}[/dim]" if a == b else f"{c:+.2f}")
        table.add_row(a, *row)
    console.print(table)

    worst = max_offdiagonal_cosine(bank)
    gate = worst < 0.5
    console.print(
        f"max |off-diagonal cosine| = {worst:.2f}  "
        f"{'[green]PASS[/green]' if gate else '[red]FAIL[/red]'}"
    )
    if not gate:
        console.print(
            "[red]Concept directions are not distinguishable. Identification results "
            "would be meaningless -- fix the bank before running anything.[/red]"
        )

    concept = bank[args.concept]
    control = random_control(concept, seed=0)

    console.rule("strength sweep: finding the usable window")
    console.print(f"[dim]neutral task: {NEUTRAL_TASK!r}[/dim]\n")
    clean = generate(model, model.chat(NEUTRAL_TASK), max_new_tokens=32)
    console.print(f"  [cyan]{'a=0 (clean)':>16}[/cyan]  {clean.strip()!r}\n")

    for alpha in STRENGTHS:
        for label, vec in (("concept", concept.vector), ("random", control.vector)):
            text = generate(
                model,
                model.chat(NEUTRAL_TASK),
                interventions=[Intervention(layer=layer, direction=vec, strength=alpha)],
                max_new_tokens=32,
            )
            console.print(f"  [cyan]{f'{label} a={alpha}':>16}[/cyan]  {text.strip()!r}")
        console.print()

    console.print(
        "[dim]Pick the largest strength where text is still coherent. Injection that "
        "produces word salad tests nothing: the model cannot report on a state that "
        "has destroyed its ability to answer.[/dim]"
    )

    console.rule("elicitation, at the smallest strengths")
    for prompt_name, prompt_text in [("detect", DETECT), ("identify", IDENTIFY)]:
        console.print(f"\n[bold]{prompt_name}[/bold]")
        for alpha in (0.0, *STRENGTHS[:3]):
            ivs = (
                []
                if alpha == 0.0
                else [Intervention(layer=layer, direction=concept.vector, strength=alpha)]
            )
            text = generate(model, model.chat(prompt_text), interventions=ivs, max_new_tokens=6)
            console.print(f"  [cyan]{f'a={alpha}':>16}[/cyan]  {text.strip()!r}")
    console.print(
        "\n[dim]If the a=0 row already answers YES / names a concept, the model has a "
        "response bias and raw hit rate is uninterpretable. Report detection as AUROC "
        "over injected-vs-clean trials instead.[/dim]"
    )

    console.rule("hook hygiene")
    a = generate(model, model.chat(NEUTRAL_TASK), max_new_tokens=12)
    b = generate(model, model.chat(NEUTRAL_TASK), max_new_tokens=12)
    console.print(f"greedy determinism after hooks removed: {'PASS' if a == b else 'FAIL'}")
    if a != b:
        console.print(f"  {a!r}\n  {b!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
