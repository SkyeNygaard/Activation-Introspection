"""Fixed-source transfer-probe propagation as a function of read depth.

This script constructs a concept bank at one injection layer, injects there, and
reads successive downstream layers with natural-text-trained probes. It answers
"how does a fixed edit propagate?" It does *not* answer "what happens when I
inject at each layer?" and must not be correlated with held-out-layer IFT
accuracy from ``run_ift.py``. That mismatched comparison produced the retracted
r = -0.774 headline.

Prior work already trains models to verbalize internal states, including IFT
([arXiv 2607.14111](https://arxiv.org/abs/2607.14111)), Steering Awareness
([arXiv 2511.21399](https://arxiv.org/abs/2511.21399)), and Introspection
Adapters ([arXiv 2604.16812](https://arxiv.org/abs/2604.16812)). The earlier
version of this docstring said these papers left this exact quantity unmeasured
and quoted open questions that the audit could not verify. Those novelty claims
are withdrawn.

This remains useful as a descriptive propagation diagnostic and as a check that
a natural-text decision boundary transfers to injected states. It does not show
causal use, introspective access, or IFT headroom. The near-output readout is
especially vulnerable to token-promotion geometry.

## Efficiency

Activations at every layer come from a single forward pass per trial via
multi-layer capture, so a full-depth profile costs the same number of forwards as
a single-layer run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from introspect.concepts import (
    DEFAULT_CONCEPTS,
    ConceptVector,
    build_bank,
    max_offdiagonal_cosine,
)
from introspect.grading import score_choices
from introspect.hooks import Intervention, capture, intervene
from introspect.metrics import accuracy
from introspect.models import DEFAULT_MODEL, LoadedModel, load
from introspect.probe import NATURAL_TEMPLATES, fit_probe_grouped
from introspect.prompts import (
    IDENTIFY_FORCED_CHOICE_VARIANTS,
    forced_choice,
    permuted_options,
    variant,
)

console = Console()


@torch.no_grad()
def collect_all_layers(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    inject_layer: int,
    strength: float,
    layers: list[int],
    n_seeds: int,
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
    list[bool],
]:
    """One forward pass per trial; activations harvested at every probe layer."""
    concepts = sorted(bank)
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
            options = permuted_options(concepts, seed)
            prompt = model.chat(
                variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(
                    options=forced_choice(options)
                )
            )
            ids = model.encode(prompt)
            # One intervened forward supplies both the captured activations and
            # the digit logits. Previously ``score_choices(..., interventions=[iv])``
            # was called while the same outer intervention hook was still live,
            # applying the edit twice on self-report trials.
            with (
                intervene(model, [iv], prompt_len=int(ids.shape[1])),
                capture(model, layers) as store,
            ):
                choice = score_choices(model, prompt, digits)
            for layer in layers:
                inj[layer].append(store.last_token(layer)[0].numpy())
            inj_y.append(idx)
            self_report.append(choice.argmax == options.index(name))

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
    ap.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="option orders per concept (nuisance repetitions, not model seeds)",
    )
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
    table = Table("layer", "depth %", "within-natural", "transfer", "permuted null", "descriptive")
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

        # Option orders and adjacent layers are not independent samples. The old
        # IID interval-separation rule overstated this as an inferential verdict.
        # Keep only the point-estimate comparison as a descriptive diagnostic.
        above_null = transfer.value > null.value
        table.add_row(
            str(layer),
            f"{100 * layer / model.n_layers:.0f}%",
            f"{within.value:.3f}",
            f"{transfer.value:.3f}",
            f"{null.value:.3f}",
            "[green]transfer > null[/green]" if above_null else "[dim]not above null[/dim]",
        )
        rows.append(
            {
                "layer": layer,
                "depth_frac": layer / model.n_layers,
                "within_natural": within.value,
                "transfer": [transfer.value, transfer.lo, transfer.hi],
                "null": [null.value, null.lo, null.hi],
                "transfer_above_null_descriptive": bool(above_null),
            }
        )

    console.print(table)
    console.print(
        f"\n[dim]Self-report is flat at {report.value:.3f} regardless of layer -- it is one "
        f"measurement of the model, not a per-layer quantity. The profile is what varies, "
        f"but neither the fixed-source profile nor its repeated option orders provide "
        f"independent inferential samples.[/dim]"
    )

    onset = next((r for r in rows if r["transfer_above_null_descriptive"]), None)
    if onset:
        console.print(
            f"\n[bold]First read layer whose point estimate exceeds this pooled null: "
            f"layer {onset['layer']} ({100 * cast(float, onset['depth_frac']):.0f}% depth). "
            f"This is a fixed-source propagation diagnostic, not an IFT failure "
            f"prediction.[/bold]"
        )

    out = args.out or Path("results") / f"layer_profile_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "descriptive_fixed_source_profile",
                "model": model.name,
                "inject_layer": inject,
                "strength": args.strength,
                "n_train": len(nat_y),
                "n_test": len(inj_y),
                "chance": chance,
                "self_report": [report.value, report.lo, report.hi],
                "inference_warning": (
                    "Intervals in nested metric objects are IID diagnostics only. Option "
                    "orders are nuisance repetitions and layers are dependent sites."
                ),
                "layers": rows,
            },
            indent=2,
        )
    )
    console.print(f"\nwrote {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
