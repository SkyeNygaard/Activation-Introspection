"""Fit the "an injection happened" direction and test whether it is real.

Step 3 of [notes/38](../notes/38-identity-or-displacement.md), which is the gate
for everything after it. If the pooled displacement direction does not separate
injected from clean states on held-out concepts, there is nothing coherent to
ablate and the rest of that design is void.

**Read downstream, not at the injection site.** At the injection layer,
injected-minus-clean is the injected direction by construction, so a separation
score there is a tautology. This reads the final residual state -- what the
network turned the edit into by the time the reporter answers.

The direction is fitted on development concepts and scored on held-out ones. That
split is declared in notes/38 and fixed here before any number exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from introspect.concepts import ConceptVector, build_bank, random_control, shuffled_control
from introspect.hooks import Intervention, capture, intervene
from introspect.models import load, loaded_revision
from introspect.preflight import check as preflight_check

#: Held fixed across conditions so the only difference is the hidden edit.
CARRIERS = (
    "A routine maintenance record is open. Hidden state marker: §",
    "A standard archive entry is visible. Hidden state marker: §",
    "The inventory log has been retrieved. Hidden state marker: §",
    "A scheduling note is displayed. Hidden state marker: §",
    "The reference sheet is now open. Hidden state marker: §",
    "A summary page has loaded. Hidden state marker: §",
)

#: Clean states are split too. The 0.5B pilot scored held-out concepts against the
#: *same* clean states used to fit, which is not a held-out comparison on the side
#: that matters. Carriers 0-2 fit, 3-5 score.
DEV_CARRIERS = slice(0, 3)
HELDOUT_CARRIERS = slice(3, 6)

#: Declared before the run. Fit on DEV, score on HELDOUT.
DEV_CONCEPTS = ["guitar", "harbor", "lantern", "meadow"]
HELDOUT_CONCEPTS = ["satellite", "teapot", "tunnel", "whale"]

ARMS = ("target", "random", "shuffled")


def _final_state(model: object, text: str, read_layer: int, ivs: list[Intervention]) -> Tensor:
    """Last-token residual at ``read_layer``, with ``ivs`` applied during the pass."""
    tok = model.tokenizer(text, return_tensors="pt").to(model.device)  # type: ignore[attr-defined]
    # Order matters: intervene registers first, so capture records the edited
    # stream. Reversing these silently records clean states under an edited label.
    with intervene(model, ivs), capture(model, [read_layer]) as cap:  # type: ignore[arg-type]
        model.model(tok.input_ids)  # type: ignore[attr-defined]
    return cap.last_token(read_layer)[0]


def _directions(bank: dict[str, ConceptVector], concept: str) -> dict[str, ConceptVector]:
    target = bank[concept]
    return {
        "target": target,
        "random": random_control(target, seed=0),
        "shuffled": shuffled_control(target, seed=0),
    }


def collect(
    model: object,
    bank: dict[str, ConceptVector],
    concepts: list[str],
    carriers: tuple[str, ...],
    *,
    inject_layer: int,
    read_layer: int,
    strength: float,
) -> tuple[Tensor, Tensor]:
    """Return (clean, injected) final states over ``carriers``."""
    clean = [_final_state(model, c, read_layer, []) for c in carriers]
    injected = []
    for concept in concepts:
        dirs = _directions(bank, concept)
        for arm in ARMS:
            for carrier in carriers:
                iv = Intervention(
                    layer=inject_layer,
                    direction=dirs[arm].vector,
                    strength=strength,
                    positions="all",
                    label=f"{concept}/{arm}",
                )
                injected.append(_final_state(model, carrier, read_layer, [iv]))
    return torch.stack(clean), torch.stack(injected)


def separation(direction: Tensor, clean: Tensor, injected: Tensor) -> dict[str, float]:
    """How well a projection onto ``direction`` tells injected from clean.

    AUROC over every clean/injected pair -- the fraction of pairs the projection
    orders correctly. 0.5 is no information; 1.0 is perfect.
    """
    unit = direction / (direction.norm() + 1e-8)
    c, i = clean @ unit, injected @ unit
    wins = (i.unsqueeze(1) > c.unsqueeze(0)).float().mean().item()
    return {
        "auroc": wins,
        "clean_mean": c.mean().item(),
        "injected_mean": i.mean().item(),
        "n_clean": int(clean.shape[0]),
        "n_injected": int(injected.shape[0]),
    }


def displacement_share(clean: Tensor, injected: Tensor) -> dict[str, float]:
    """How much of the injected-minus-clean signal is *shared* across injections.

    notes/38 names this as the interpretability bound on the whole design: a
    rank-1 ablation can only remove the shared part, so if that part is small,
    "the reports survived ablation" means almost nothing.

    Two numbers, because the 0.5B pilot showed one is not enough:

    * ``mean_share`` -- energy along the mean delta. This is what a rank-1
      ablation of the pooled direction actually removes.
    * ``first_pc_share`` -- energy along the leading component. Higher than
      ``mean_share`` means the dominant axis is concept-specific structure
      rather than the shared "something happened" offset.
    """
    # Not mean-centred: the shared offset between injected and clean is the
    # quantity of interest, and centring subtracts exactly that.
    deltas = injected - clean.mean(0, keepdim=True)
    total = float((deltas**2).sum().item())

    mean_delta = deltas.mean(0)
    unit = mean_delta / (mean_delta.norm() + 1e-8)
    along_mean = float(((deltas @ unit) ** 2).sum().item())

    sv = torch.linalg.svdvals(deltas)
    return {
        "mean_share": along_mean / (total + 1e-12),
        "first_pc_share": float((sv[0] ** 2 / (sv**2).sum()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--inject-layer", type=int, required=True)
    parser.add_argument("--read-layer", type=int, default=-1, help="-1 = final block")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new path rather than overwriting")
    preflight_check(args.model, training=False)

    model = load(args.model)
    read_layer = args.read_layer if args.read_layer >= 0 else len(model.blocks) - 1
    bank = build_bank(model, args.inject_layer, DEV_CONCEPTS + HELDOUT_CONCEPTS)

    kw = dict(inject_layer=args.inject_layer, read_layer=read_layer, strength=args.strength)
    dev = CARRIERS[DEV_CARRIERS]
    out = CARRIERS[HELDOUT_CARRIERS]
    dev_clean, dev_injected = collect(model, bank, DEV_CONCEPTS, dev, **kw)  # type: ignore[arg-type]
    out_clean, out_injected = collect(model, bank, HELDOUT_CONCEPTS, out, **kw)  # type: ignore[arg-type]

    direction = dev_injected.mean(0) - dev_clean.mean(0)

    result = {
        "model": args.model,
        "model_revision": loaded_revision(model),
        "inject_layer": args.inject_layer,
        "read_layer": read_layer,
        "strength": args.strength,
        "n_blocks": len(model.blocks),
        "dev_concepts": DEV_CONCEPTS,
        "heldout_concepts": HELDOUT_CONCEPTS,
        "dev_carriers": list(dev),
        "heldout_carriers": list(out),
        "fit": separation(direction, dev_clean, dev_injected),
        "heldout": separation(direction, out_clean, out_injected),
        "displacement_share_heldout": displacement_share(out_clean, out_injected),
        "gate": (
            "notes/38: heldout auroc near 0.5 means stop. "
            "low mean_share bounds every downstream ablation result"
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def _self_check() -> None:
    """The scoring is the only non-obvious logic here, so pin it."""
    torch.manual_seed(0)
    d = torch.zeros(8)
    d[0] = 1.0
    clean = torch.randn(20, 8) * 0.1
    injected = clean.clone()[:20] + d * 5.0
    assert separation(d, clean, injected)["auroc"] == 1.0
    assert abs(separation(d, clean, clean)["auroc"] - 0.5) < 0.15
    noise = torch.randn(20, 8)
    assert displacement_share(clean, clean + d * 50.0)["mean_share"] > 0.9
    assert displacement_share(clean, clean + noise)["mean_share"] < 0.5
    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
