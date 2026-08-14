"""Train two reporters: one normal, one that cannot see that anything was injected.

The experiment [notes/38](../notes/38-identity-or-displacement.md) was written for.
An earlier run established that at the readout one direction carries all of "an
injection happened" and none of "which concept" -- removing it drops
injected-versus-clean discrimination to chance while leaving concept identity
untouched. This trains a reporter with that direction removed on every forward
pass, and compares it against an identically trained reporter without the removal.

* Reports survive -> the adapter reads concept identity.
* Reports collapse -> it was reading disturbance, and the concept vocabulary rode
  along on whatever the training paired with it.

The ablation is applied by wrapping training and evaluation in an outer
``intervene`` block. Forward hooks compose, so the concept injection inside
``ift.train`` still fires; nothing in the hash-bound training module is touched.

Nothing here is a claim about introspection in general. It is one model, one
injection site, one training recipe.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from introspect import ift
from introspect.concepts import build_bank
from introspect.hooks import Intervention, intervene
from introspect.models import load, loaded_revision
from introspect.preflight import check as preflight_check

#: Short name -> repo. Kept out of src/introspect/models.py on purpose: that file
#: is hashed into frozen protocols and adding a key there breaks them.
MODEL_REPOS = {"qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507"}

#: Disjoint from the evaluation seeds, so the adapter is scored on prompt
#: paraphrases it never trained on as well as on concepts it never saw.
TRAIN_SEEDS = tuple(range(0, 6))
EVAL_SEEDS = tuple(range(100, 106))


def _displacement_direction(
    model: object, bank: dict[str, Any], inject_layer: int, read_layer: int, strength: float
) -> Tensor:
    """Refit the shared "an injection happened" direction, on development rows only.

    Imported from the sibling script rather than duplicated so the two cannot
    drift apart -- it is the same fit that produced the numbers in notes/38.
    """
    path = Path(__file__).with_name("run_displacement_direction.py")
    spec = importlib.util.spec_from_file_location("_disp", path)
    mod = importlib.util.module_from_spec(cast(Any, spec))
    cast(Any, spec).loader.exec_module(mod)
    clean, injected, _ = mod.collect(
        model,
        bank,
        mod.DEV_CONCEPTS,
        mod.DEV_CARRIERS,
        inject_layer=inject_layer,
        read_layer=read_layer,
        strength=strength,
    )
    return cast(Tensor, injected.mean(0) - clean.mean(0))


def _free(model: object) -> None:
    """The machine holds one model at a time; two arms means loading twice."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_arm(
    model_name: str,
    *,
    ablate: bool,
    direction: Tensor | None,
    inject_layer: int,
    read_layer: int,
    strength: float,
    epochs: int,
    train_concepts: list[str],
    eval_concepts: list[str],
) -> dict[str, object]:
    """Train one adapter and score it on held-out concepts."""
    model = load(MODEL_REPOS.get(model_name, model_name))
    ift.attach_lora(model)

    full = build_bank(model, inject_layer, train_concepts + eval_concepts)
    train_bank = {k: v for k, v in full.items() if k in train_concepts}
    eval_bank = {k: v for k, v in full.items() if k in eval_concepts}

    extra: list[Intervention] = []
    if ablate:
        assert direction is not None
        extra = [
            Intervention(
                layer=read_layer,
                direction=direction,
                mode="ablate",
                positions="last",
                label="ablate_displacement",
            )
        ]

    examples = ift.build_examples(model, train_bank, [inject_layer], [strength], seeds=TRAIN_SEEDS)
    digits = ift.digit_token_ids(model, len(train_bank))

    # Outer hook: present on every forward pass in both training and evaluation.
    with intervene(model, extra):
        losses = ift.train(model, examples, digits, epochs=epochs, seed=0)
        correct = ift.evaluate_layer(model, eval_bank, inject_layer, strength, seeds=EVAL_SEEDS)

    _free(model)
    return {
        "ablated": ablate,
        "n_train_examples": len(examples),
        "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
        "heldout_accuracy": sum(correct) / len(correct),
        "n_eval": len(correct),
        "chance": 1.0 / len(eval_bank),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen-3b")
    p.add_argument("--inject-layer", type=int, default=9)
    p.add_argument("--read-layer", type=int, default=-1)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new path")
    preflight_check(args.model, training=True)

    # Fit the direction once, on its own model load, so the two training arms are
    # identical apart from the ablation.
    probe = load(MODEL_REPOS.get(args.model, args.model))
    read_layer = args.read_layer if args.read_layer >= 0 else len(probe.blocks) - 1
    probe_bank = build_bank(probe, args.inject_layer)
    direction = _displacement_direction(
        probe, probe_bank, args.inject_layer, read_layer, args.strength
    )
    revision = loaded_revision(probe)
    _free(probe)

    train_concepts = [*sorted(probe_bank)[:4], "guitar", "harbor", "lantern", "meadow"]
    eval_concepts = ["satellite", "teapot", "tunnel", "whale"]
    kw = dict(
        direction=direction,
        inject_layer=args.inject_layer,
        read_layer=read_layer,
        strength=args.strength,
        epochs=args.epochs,
        train_concepts=train_concepts,
        eval_concepts=eval_concepts,
    )

    arms = {}
    for name, ablate in (("plain", False), ("ablated", True)):
        print(f"=== arm: {name} ===", flush=True)
        arms[name] = run_arm(args.model, ablate=ablate, **kw)

    result = {
        "model": args.model,
        "model_revision": revision,
        "inject_layer": args.inject_layer,
        "read_layer": read_layer,
        "strength": args.strength,
        "epochs": args.epochs,
        "train_concepts": train_concepts,
        "eval_concepts": eval_concepts,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "arms": arms,
        "reading": (
            "ablated close to plain -> the adapter reads concept identity. "
            "ablated at chance -> it was reading that something was disturbed."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
