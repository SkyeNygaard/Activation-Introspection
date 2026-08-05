"""Measure where single-layer injection-report fine-tuning transfers.

Train the model to report injections at ONE layer, then measure accuracy at every
other layer. Every held-out site receives a concept bank constructed at that
same site; cross-layer use of the training bank is rejected.

The old comparison to ``layer_profile.py`` is intentionally unavailable. That
profile injects at one fixed source layer and reads at successively later layers,
whereas this script injects at each evaluated layer. Correlating the two creates
opposite depth trends by construction. Only a profile produced by
``run_reach_output.py`` (inject at L, read at output) is site-matched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from introspect.artifacts import load_matched_profile
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
    ap.add_argument(
        "--train-seeds", type=int, default=6, help="option orders per training paraphrase"
    )
    ap.add_argument(
        "--eval-seeds", type=int, default=15, help="option orders per evaluation paraphrase"
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--training-seed", type=int, default=0)
    ap.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="RETIRED: fixed-source layer_profile files are not site-matched",
    )
    ap.add_argument(
        "--matched-profile",
        type=Path,
        default=None,
        help="site-matched summary from scripts/run_reach_output.py",
    )
    ap.add_argument(
        "--allow-unpinned-model-profile",
        action="store_true",
        help=(
            "exploratory only: allow a profile join when the model revision is absent; "
            "immutable weight identity cannot then be verified"
        ),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.profile is not None:
        ap.error(
            "--profile is retired: layer_profile.py injects at a fixed source and is not "
            "comparable to inject-at-L IFT evaluation. Generate a matched profile with "
            "scripts/run_reach_output.py and pass --matched-profile instead."
        )

    model = load(args.model)
    config = getattr(model.model, "config", None)
    model_revision = getattr(config, "_commit_hash", None)
    matched_profile = (
        load_matched_profile(
            args.matched_profile,
            expected_model=model.name,
            expected_strength=args.strength,
            expected_concepts=list(DEFAULT_CONCEPTS),
            expected_n_layers=model.n_layers,
            expected_model_revision=model_revision,
            require_model_revision=not args.allow_unpinned_model_profile,
        )
        if args.matched_profile
        else {}
    )
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
    # confounds distance from the trained layer with absolute depth.
    candidate_layers = [layer for layer in range(1, model.n_layers) if layer != train_layer]
    console.print("building a pre-IFT concept bank at every evaluation site", style="dim")
    eval_banks = {
        layer: build_bank(model, layer, concepts=DEFAULT_CONCEPTS) for layer in candidate_layers
    }
    bank_cosines = {layer: max_offdiagonal_cosine(b) for layer, b in eval_banks.items()}
    skipped_layers = [layer for layer, cosine in bank_cosines.items() if cosine >= 0.5]
    eval_layers = [layer for layer in candidate_layers if layer not in skipped_layers]
    if skipped_layers:
        console.print(
            f"[yellow]skipping degenerate site-specific banks at layers {skipped_layers}[/yellow]"
        )
    if not eval_layers:
        console.print("[red]No non-degenerate held-out layer banks; aborting.[/red]")
        return
    digit_ids = digit_token_ids(model, len(bank))

    # Disjoint paraphrase sets: the adapter is scored on wordings it never saw.
    train_seeds = seeds_for(TRAIN_VARIANTS, args.train_seeds)
    eval_seeds = seeds_for(EVAL_VARIANTS, args.eval_seeds)

    console.rule("before training")
    pre = {
        layer: accuracy(
            evaluate_layer(
                model,
                eval_banks[layer],
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
    # This seed must be set before adapter construction: LoRA initialization and
    # training-time dropout are part of an independent training run, not merely
    # the example-order shuffle inside ``train``.
    torch.manual_seed(args.training_seed)
    attach_lora(model)
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    console.print(f"LoRA trainable params: {trainable / 1e6:.2f}M")

    examples = build_examples(model, bank, [train_layer], [args.strength], seeds=train_seeds)
    console.print(f"{len(examples)} training examples, {args.epochs} epochs")
    losses = train(
        model,
        examples,
        digit_ids,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.training_seed,
    )
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
                eval_banks[layer],
                layer,
                args.strength,
                seeds=eval_seeds,
            )
        )
        for layer in eval_layers
    }

    chance = 1 / len(bank)
    table = Table(
        "layer",
        "depth",
        "dist from trained",
        "pre-IFT",
        "post-IFT",
        "gain",
        "matched reach-to-output",
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
            f"{matched_profile.get(layer, float('nan')):.3f}" if matched_profile else "-",
        )
        rows.append(
            {
                "layer": layer,
                "depth_frac": layer / model.n_layers,
                "distance": layer - train_layer,
                "pre_accuracy": pre[layer].value,
                "post_accuracy": post[layer].value,
                "n_option_order_evaluations": post[layer].n,
                "gain": gain,
                "matched_reach_output": matched_profile.get(layer),
                "bank_max_abs_offdiag_cosine": bank_cosines[layer],
            }
        )
    console.print(table)
    console.print(f"[dim]chance = {chance:.3f}[/dim]")

    if matched_profile:
        matched_rows = [r for r in rows if r["matched_reach_output"] is not None]
        xs = np.array([r["matched_reach_output"] for r in matched_rows])
        ys = np.array([r["post_accuracy"] for r in matched_rows])
        gains = np.array([r["gain"] for r in matched_rows])
        if len(xs) > 2 and xs.std() > 0:
            r_post = float(np.corrcoef(xs, ys)[0, 1])
            r_gain = float(np.corrcoef(xs, gains)[0, 1])
            console.rule("site-matched descriptive comparison")
            console.print(
                f"correlation, reach-to-output vs post-IFT accuracy: "
                f"[bold]r = {r_post:+.3f}[/bold]  (n={len(xs)} held-out layers)"
            )
            console.print(
                f"correlation, reach-to-output vs IFT gain:             r = {r_gain:+.3f}"
            )
            console.print(
                "\n[dim]Descriptive only: layers are ordered, dependent sites and do not "
                "supply independent correlation samples.[/dim]"
            )

    out = args.out or Path("results") / f"ift_{args.model}_L{train_layer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "exploratory_single_adapter_run",
                "model": model.name,
                "model_revision": model_revision,
                "train_layer": train_layer,
                "strength": args.strength,
                "chance": chance,
                "n_train_examples": len(examples),
                "train_option_orders_per_variant": args.train_seeds,
                "eval_option_orders_per_variant": args.eval_seeds,
                "train_variants": list(TRAIN_VARIANTS),
                "eval_variants": list(EVAL_VARIANTS),
                "training_seed": args.training_seed,
                "evaluation_mode": "dropout_disabled",
                "vector_bank_policy": "constructed_at_each_injection_layer",
                "skipped_degenerate_layers": skipped_layers,
                "matched_profile": str(args.matched_profile) if args.matched_profile else None,
                "matched_profile_requires_model_revision": not args.allow_unpinned_model_profile,
                "inference_warning": (
                    "Point estimates describe one adapter run. Option orders and repeated "
                    "prompt variants are nuisance evaluations, not independent samples; "
                    "no confidence or equivalence interval is reported. Raw IFT trial "
                    "records and multi-run orchestration remain required for confirmation."
                ),
                "trained_layer_pre": pre_train_layer.value,
                "trained_layer_post": post_train_layer.value,
                "layers": rows,
            },
            indent=2,
        )
    )
    console.print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
