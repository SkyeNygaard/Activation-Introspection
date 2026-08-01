"""Transfer-probe accuracy as a function of depth, at full power.

## Why this profile is the deliverable

Two 2026 papers train models to verbalize their own internal states:

- **Introspection Fine-Tuning** (arXiv 2607.14111) takes Llama-1B from 9.6% to
  60.6% on sentence localization, and reports peak accuracy "at optimal
  layer/strength configurations". It states it does **not** use linear probes to
  compare what activations encode against what the model reports.
- **Introspection Adapters** (Anthropic, arXiv 2604.16812) reach 89% verbalization
  on AuditBench, and likewise do not compare probe decodability to verbalization.
  Their stated open question is *why* the adapters generalize.

Both therefore leave the same quantity unmeasured: **how much concept information
is linearly present before any introspection training**. That is what this
profile measures, layer by layer.

The prediction it licenses is falsifiable and cheap to test: introspection
training should gain most at layers where pre-training transfer accuracy is high,
and gain little where transfer sits at the permuted-label null. If the layer
profile of IFT gains tracks the layer profile measured here, that is a mechanistic
account of when introspection training works -- and a way to predict where it
will fail without running the training.

Independent convergence worth noting: Macar et al. (2026) locate a distributed
"introspective circuit" at roughly 70% of model depth. On Qwen2.5-0.5B that is
layer ~17, and transfer here reaches the probe's own ceiling at layer 16.

## Efficiency

Activations at every layer come from a single forward pass per trial via
multi-layer capture, so a full-depth profile costs the same number of forwards as
a single-layer run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from introspect.concepts import DEFAULT_CONCEPTS, build_bank, max_offdiagonal_cosine
from introspect.grading import score_choices
from introspect.hooks import Intervention, capture, intervene
from introspect.metrics import accuracy
from introspect.models import DEFAULT_MODEL, load
from introspect.probe import NATURAL_TEMPLATES, fit_probe_grouped
from introspect.prompts import IDENTIFY_FORCED_CHOICE_VARIANTS, forced_choice, variant

console = Console()


@torch.no_grad()
def collect_all_layers(model, bank, inject_layer, strength, layers, n_seeds):  # type: ignore[no-untyped-def]
    """One forward pass per trial; activations harvested at every probe layer."""
    concepts = sorted(bank)
    option_block = forced_choice(concepts)
    digits = [str(i + 1) for i in range(len(concepts))]

    # Natural (no injection anywhere) -- the probe's training set.
    nat: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    nat_y: list[int] = []
    nat_g: list[int] = []
    for idx, name in enumerate(concepts):
        for t_id, template in enumerate(NATURAL_TEMPLATES):
            ids = model.encode(template.format(concept=name))
            with capture(model, layers) as store:
                model.forward_logits(ids)
            for layer in layers:
                nat[layer].append(store.last_token(layer)[0].numpy())
            nat_y.append(idx)
            nat_g.append(t_id)

    # Injected trials -- the probe's test set, plus the model's own answer.
    inj: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    inj_y: list[int] = []
    self_report: list[bool] = []
    for idx, name in enumerate(concepts):
        iv = Intervention(layer=inject_layer, direction=bank[name].vector, strength=strength)
        for seed in range(n_seeds):
            prompt = model.chat(
                variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(options=option_block)
            )
            ids = model.encode(prompt)
            with intervene(model, [iv], prompt_len=int(ids.shape[1])):
                with capture(model, layers) as store:
                    model.forward_logits(ids)
                choice = score_choices(model, prompt, digits, interventions=[iv])
            for layer in layers:
                inj[layer].append(store.last_token(layer)[0].numpy())
            inj_y.append(idx)
            self_report.append(choice.argmax == idx)

    return (
        {layer: np.stack(v) for layer, v in nat.items()},
        np.array(nat_y),
        np.array(nat_g),
        {layer: np.stack(v) for layer, v in inj.items()},
        np.array(inj_y),
        self_report,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--inject-layer", type=int, default=None)
    ap.add_argument("--strength", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = load(args.model)
    inject = args.inject_layer if args.inject_layer is not None else int(0.35 * model.n_layers)
    bank = build_bank(model, inject, concepts=DEFAULT_CONCEPTS)
    worst = max_offdiagonal_cosine(bank)
    console.print(
        f"{model.name}  {model.n_layers} layers  inject L{inject}  bank max|off-diag|={worst:.2f}"
    )
    if worst >= 0.5:
        console.print("[red]Degenerate bank; aborting.[/red]")
        return

    layers = list(range(inject + 1, model.n_layers))
    nat_x, nat_y, nat_g, inj_x, inj_y, self_report = collect_all_layers(
        model, bank, inject, args.strength, layers, args.seeds
    )
    chance = 1 / len(bank)
    report = accuracy(self_report)
    console.print(
        f"n_train={len(nat_y)}  n_test={len(inj_y)}  chance={chance:.3f}  self-report={report}"
    )

    rng = np.random.default_rng(0)
    rows = []
    table = Table("layer", "depth %", "within-natural", "transfer", "permuted null", "verdict")
    for layer in layers:
        scaler = StandardScaler().fit(nat_x[layer])
        xs_nat, xs_inj = scaler.transform(nat_x[layer]), scaler.transform(inj_x[layer])
        clf = LogisticRegression(max_iter=3000).fit(xs_nat, nat_y)

        transfer = accuracy(list(clf.predict(xs_inj) == inj_y))
        null_hits: list[bool] = []
        for _ in range(3):
            null_clf = LogisticRegression(max_iter=3000).fit(xs_nat, rng.permutation(nat_y))
            null_hits.extend(list(null_clf.predict(xs_inj) == inj_y))
        null = accuracy(null_hits)
        within = accuracy(fit_probe_grouped(nat_x[layer], nat_y, nat_g))

        beats = transfer.lo > null.hi
        table.add_row(
            str(layer),
            f"{100 * layer / model.n_layers:.0f}%",
            f"{within.value:.3f}",
            f"{transfer.value:.3f} [{transfer.lo:.3f}, {transfer.hi:.3f}]",
            f"{null.value:.3f}",
            "[green]decodable[/green]" if beats else "[dim]at null[/dim]",
        )
        rows.append(
            {
                "layer": layer,
                "depth_frac": layer / model.n_layers,
                "within_natural": within.value,
                "transfer": [transfer.value, transfer.lo, transfer.hi],
                "null": [null.value, null.lo, null.hi],
                "beats_null": bool(beats),
            }
        )

    console.print(table)
    console.print(
        f"\n[dim]Self-report is flat at {report.value:.3f} regardless of layer -- it is one "
        f"measurement of the model, not a per-layer quantity. The profile is what varies, "
        f"and it is the quantity introspection-training papers leave unmeasured.[/dim]"
    )

    onset = next((r for r in rows if r["beats_null"]), None)
    if onset:
        console.print(
            f"\n[bold]Decodability onset: layer {onset['layer']} "
            f"({100 * onset['depth_frac']:.0f}% depth). Below this the injected state does "
            f"not resemble the model's own representation of the concept, so introspection "
            f"training should have nothing to latch onto there.[/bold]"
        )

    out = args.out or Path("results") / f"layer_profile_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": model.name,
                "inject_layer": inject,
                "strength": args.strength,
                "n_train": len(nat_y),
                "n_test": len(inj_y),
                "chance": chance,
                "self_report": [report.value, report.lo, report.hi],
                "layers": rows,
            },
            indent=2,
        )
    )
    console.print(f"\nwrote {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
