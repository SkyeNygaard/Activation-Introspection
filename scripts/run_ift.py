"""Does the transfer-probe profile predict where introspection training generalizes?

Train the model to report injections at ONE layer, then measure accuracy at every
other layer. The prediction under test: held-out-layer accuracy tracks the
pre-training transfer-probe profile from `layer_profile.py`.

This is the falsifiable form of the claim, and it addresses the open question IFT
(arXiv 2607.14111) states about layer-agnostic generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from introspect.concepts import DEFAULT_CONCEPTS, build_bank, max_offdiagonal_cosine
from introspect.ift import (
    EVAL_VARIANTS,
    TRAIN_VARIANTS,
    attach_lora,
    build_examples,
    digit_token_ids,
    evaluate_layer,
    seeds_for,
    train,
)
from introspect.metrics import accuracy
from introspect.models import DEFAULT_MODEL, load

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--inject-layer", type=int, default=None, help="layer trained on")
    ap.add_argument("--strength", type=float, default=0.2)
    ap.add_argument("--train-seeds", type=int, default=6, help="repeats per training paraphrase")
    ap.add_argument("--eval-seeds", type=int, default=15, help="repeats per eval paraphrase")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--profile", type=Path, default=None, help="layer_profile json to compare")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = load(args.model)
    train_layer = args.inject_layer if args.inject_layer is not None else int(0.35 * model.n_layers)
    bank = build_bank(model, train_layer, concepts=DEFAULT_CONCEPTS)
    worst = max_offdiagonal_cosine(bank)
    console.print(
        f"{model.name}  {model.n_layers} layers  train on L{train_layer}  "
        f"bank max|off-diag|={worst:.2f}"
    )
    if worst >= 0.5:
        console.print("[red]Degenerate bank; aborting.[/red]")
        return

    # Evaluate on BOTH sides of the training layer. Testing only downstream
    # confounds "distance from the trained layer" with "absolute depth" -- the
    # two hypotheses this experiment exists to separate.
    eval_layers = [layer for layer in range(1, model.n_layers) if layer != train_layer]
    digit_ids = digit_token_ids(model, len(bank))

    # Disjoint paraphrase sets: the adapter is scored on wordings it never saw.
    train_seeds = seeds_for(TRAIN_VARIANTS, args.train_seeds)
    eval_seeds = seeds_for(EVAL_VARIANTS, args.eval_seeds)

    console.rule("before training")
    pre = {
        layer: accuracy(
            evaluate_layer(
                model,
                bank,
                layer,
                args.strength,
                seeds=eval_seeds,
            )
        )
        for layer in eval_layers
    }
    pre_train_layer = accuracy(
        evaluate_layer(
            model,
            bank,
            train_layer,
            args.strength,
            seeds=eval_seeds,
        )
    )
    console.print(f"trained layer L{train_layer}: {pre_train_layer}")
    console.print(f"mean over held-out layers: {np.mean([e.value for e in pre.values()]):.3f}")

    console.rule("training")
    attach_lora(model)
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    console.print(f"LoRA trainable params: {trainable / 1e6:.2f}M")

    examples = build_examples(model, bank, [train_layer], [args.strength], seeds=train_seeds)
    console.print(f"{len(examples)} training examples, {args.epochs} epochs")
    losses = train(model, examples, digit_ids, epochs=args.epochs, lr=args.lr)
    console.print(f"loss {np.mean(losses[:20]):.4f} -> {np.mean(losses[-20:]):.4f}")

    console.rule("after training")
    post_train_layer = accuracy(
        evaluate_layer(
            model,
            bank,
            train_layer,
            args.strength,
            seeds=eval_seeds,
        )
    )
    post = {
        layer: accuracy(
            evaluate_layer(
                model,
                bank,
                layer,
                args.strength,
                seeds=eval_seeds,
            )
        )
        for layer in eval_layers
    }

    profile = {}
    if args.profile and args.profile.exists():
        data = json.loads(args.profile.read_text())
        profile = {r["layer"]: r["transfer"][0] for r in data["layers"]}

    chance = 1 / len(bank)
    table = Table(
        "layer", "depth", "dist from trained", "pre-IFT", "post-IFT", "gain", "probe transfer"
    )
    table.add_row(
        f"[bold]{train_layer}[/bold]",
        f"{100 * train_layer / model.n_layers:.0f}%",
        "0",
        f"{pre_train_layer.value:.3f}",
        f"{post_train_layer.value:.3f}",
        f"{post_train_layer.value - pre_train_layer.value:+.3f}",
        "[dim]trained[/dim]",
    )
    rows = []
    for layer in eval_layers:
        gain = post[layer].value - pre[layer].value
        table.add_row(
            str(layer),
            f"{100 * layer / model.n_layers:.0f}%",
            f"{layer - train_layer:+d}",
            f"{pre[layer].value:.3f}",
            f"{post[layer].value:.3f}",
            f"{gain:+.3f}",
            f"{profile.get(layer, float('nan')):.3f}" if profile else "-",
        )
        rows.append(
            {
                "layer": layer,
                "depth_frac": layer / model.n_layers,
                "distance": layer - train_layer,
                "pre": [pre[layer].value, pre[layer].lo, pre[layer].hi],
                "post": [post[layer].value, post[layer].lo, post[layer].hi],
                "gain": gain,
                "probe_transfer": profile.get(layer),
            }
        )
    console.print(table)
    console.print(f"[dim]chance = {chance:.3f}[/dim]")

    if profile:
        xs = np.array([r["probe_transfer"] for r in rows if r["probe_transfer"] is not None])
        ys = np.array([r["post"][0] for r in rows if r["probe_transfer"] is not None])
        gains = np.array([r["gain"] for r in rows if r["probe_transfer"] is not None])
        if len(xs) > 2 and xs.std() > 0:
            r_post = float(np.corrcoef(xs, ys)[0, 1])
            r_gain = float(np.corrcoef(xs, gains)[0, 1])
            dist = np.array(
                [abs(r["distance"]) for r in rows if r["probe_transfer"] is not None],
                dtype=float,
            )
            r_dist = float(np.corrcoef(dist, ys)[0, 1])
            console.rule("the prediction")
            console.print(
                f"correlation, pre-training probe transfer vs post-IFT accuracy: "
                f"[bold]r = {r_post:+.3f}[/bold]  (n={len(xs)} held-out layers)"
            )
            console.print(
                f"correlation, probe transfer vs IFT gain:              r = {r_gain:+.3f}"
            )
            console.print(
                f"correlation, |distance from trained layer| vs post-IFT: "
                f"[bold]r = {r_dist:+.3f}[/bold]"
            )
            console.print(
                "\n[dim]The prediction was that pre-training decodability forecasts where "
                "introspection training generalizes. A strong positive r supports it; r near "
                "zero falsifies it.[/dim]"
            )

    out = args.out or Path("results") / f"ift_{args.model}_L{train_layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": model.name,
                "train_layer": train_layer,
                "strength": args.strength,
                "chance": chance,
                "n_train_examples": len(examples),
                "eval_seeds": args.eval_seeds,
                "trained_layer_pre": pre_train_layer.value,
                "trained_layer_post": post_train_layer.value,
                "layers": rows,
            },
            indent=2,
        )
    )
    console.print(f"\nwrote {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
