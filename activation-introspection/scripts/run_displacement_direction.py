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
    "The status board has been refreshed. Hidden state marker: §",
    "A filing index is on screen. Hidden state marker: §",
    "The daily register is showing. Hidden state marker: §",
    "A checklist window has opened. Hidden state marker: §",
    "The ledger view is active. Hidden state marker: §",
    "A dispatch summary is present. Hidden state marker: §",
)

#: Clean states are split too. The 0.5B pilot scored held-out concepts against the
#: *same* clean states used to fit, which is not a held-out comparison on the side
#: that matters.
#:
#: Widened from three per side to six after the first sweep: clean states vary only
#: by carrier, so three of them made the post-ablation separation a three-point
#: estimate, and it wandered between 0.338 and 0.588 across strengths when all
#: three values should have been chance.
DEV_CARRIERS = CARRIERS[0:6]
HELDOUT_CARRIERS = CARRIERS[6:12]

#: Declared before the run. Fit on DEV, score on HELDOUT. Widened from four each
#: to eight after the first Qwen3-4B run: with four concepts, identity decoding sat
#: at 1.000 before ablation, so "identity survives the ablation" was a claim made
#: at the ceiling and a small loss could not have been seen. Eight puts chance at
#: 0.125 and leaves room to fall.
DEV_CONCEPTS = ["guitar", "harbor", "lantern", "meadow", "satellite", "teapot", "tunnel", "whale"]
HELDOUT_CONCEPTS = [
    "ocean",
    "bread",
    "volcano",
    "violin",
    "spider",
    "hospital",
    "desert",
    "clock",
]

ARMS = ("target", "random", "shuffled")

#: Short name -> repo, kept here rather than in models.py: see the note in main().
MODEL_REPOS = {"qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507"}


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
) -> tuple[Tensor, Tensor, list[tuple[str, str, int]]]:
    """Return (clean, injected, labels) over ``carriers``. Labels are (concept, arm, carrier)."""
    clean = [_final_state(model, c, read_layer, []) for c in carriers]
    injected = []
    labels: list[tuple[str, str, int]] = []
    for concept in concepts:
        dirs = _directions(bank, concept)
        for arm in ARMS:
            for ci, carrier in enumerate(carriers):
                iv = Intervention(
                    layer=inject_layer,
                    direction=dirs[arm].vector,
                    strength=strength,
                    positions="all",
                    label=f"{concept}/{arm}",
                )
                injected.append(_final_state(model, carrier, read_layer, [iv]))
                labels.append((concept, arm, ci))
    return torch.stack(clean), torch.stack(injected), labels


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


def refit_separation(
    dev_clean: Tensor, dev_injected: Tensor, out_clean: Tensor, out_injected: Tensor
) -> float:
    """Refit a direction on development states, score on held-out ones.

    Fitting and scoring on the same rows is worthless here: with a few dozen
    points in a couple of thousand dimensions almost any two groups are linearly
    separable, so a perfect score would measure the dimensionality, not the
    signal. The refit must be held out on both sides to mean anything.
    """
    refit = dev_injected.mean(0) - dev_clean.mean(0)
    return separation(refit, out_clean, out_injected)["auroc"]


def project_out(states: Tensor, direction: Tensor) -> Tensor:
    """Remove the component along ``direction``. Same operation Intervention.ablate does."""
    unit = direction / (direction.norm() + 1e-8)
    out: Tensor = states - (states @ unit).unsqueeze(-1) * unit
    return out


def spectrum(clean: Tensor, injected: Tensor, thresholds: tuple[float, ...]) -> dict[str, float]:
    """Components needed to reach each fraction of the injected-minus-clean energy.

    Reported for completeness, but note it does *not* set the ablation rank: the
    leading components span the concept-identity variation as well as the shared
    disturbance, so ablating them would remove the signal the experiment needs to
    keep. The mean direction is the shared part by construction.
    """
    deltas = injected - clean.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(deltas)
    energy = sv**2
    cum = torch.cumsum(energy, 0) / energy.sum()
    return {f"rank_at_{t}": int((cum < t).sum().item()) + 1 for t in thresholds}


def concept_decodability(injected: Tensor, labels: list[tuple[str, str, int]]) -> float:
    """Leave-one-carrier-out nearest-centroid accuracy on the target arm.

    Does the state still say *which* concept was injected? Chance is 1/n_concepts.
    Uses only the target arm: random and shuffled directions carry no concept.
    """
    rows = [(i, c, ci) for i, (c, arm, ci) in enumerate(labels) if arm == "target"]
    concepts = sorted({c for _, c, _ in rows})
    carriers = sorted({ci for _, _, ci in rows})
    correct = total = 0
    for held in carriers:
        train = [(i, c) for i, c, ci in rows if ci != held]
        centroids = {
            c: torch.stack([injected[i] for i, cc in train if cc == c]).mean(0) for c in concepts
        }
        for i, c, ci in rows:
            if ci != held:
                continue
            best = min(concepts, key=lambda k: float((injected[i] - centroids[k]).norm()))
            correct += int(best == c)
            total += 1
    return correct / max(total, 1)


def evaluate(
    model: object,
    bank: dict[str, ConceptVector],
    *,
    inject_layer: int,
    read_layer: int,
    strength: float,
) -> dict[str, object]:
    """One strength: fit the direction on dev, then run both gates on held-out."""
    dev = DEV_CARRIERS
    out = HELDOUT_CARRIERS
    dev_clean, dev_injected, _ = collect(
        model,
        bank,
        DEV_CONCEPTS,
        dev,
        inject_layer=inject_layer,
        read_layer=read_layer,
        strength=strength,
    )
    out_clean, out_injected, out_labels = collect(
        model,
        bank,
        HELDOUT_CONCEPTS,
        out,
        inject_layer=inject_layer,
        read_layer=read_layer,
        strength=strength,
    )

    direction = dev_injected.mean(0) - dev_clean.mean(0)

    # Removing the shared direction should destroy the ability to tell injected
    # from clean, while leaving *which* concept was injected intact. If both
    # survive, the direction is not carrying the disturbance. If both die, the
    # disturbance and the identity are not separable and the design is void.
    abl = [project_out(t, direction) for t in (dev_clean, dev_injected, out_clean, out_injected)]

    return {
        "strength": strength,
        "heldout": separation(direction, out_clean, out_injected),
        "displacement_share_heldout": displacement_share(out_clean, out_injected),
        "spectrum_heldout": spectrum(out_clean, out_injected, (0.8, 0.9, 0.95)),
        "ablation": {
            "injected_vs_clean_auroc_before": separation(direction, out_clean, out_injected)[
                "auroc"
            ],
            "injected_vs_clean_auroc_after": refit_separation(abl[0], abl[1], abl[2], abl[3]),
            "concept_accuracy_before": concept_decodability(out_injected, out_labels),
            "concept_accuracy_after": concept_decodability(abl[3], out_labels),
            "concept_chance": 1.0 / len(HELDOUT_CONCEPTS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--inject-layer", type=int, required=True)
    parser.add_argument("--read-layer", type=int, default=-1, help="-1 = final block")
    parser.add_argument("--strengths", default="1.0", help="comma-separated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new path rather than overwriting")
    preflight_check(args.model, training=False)

    # Loaded by repo id rather than a registry entry on purpose. src/introspect/
    # models.py is hashed into 24 frozen protocols, and adding a KNOWN_MODELS key
    # for qwen3-4b broke the two tests that verify those hashes. The registry is
    # not worth a provenance break; load() already falls back to the raw name.
    model = load(MODEL_REPOS.get(args.model, args.model))
    read_layer = args.read_layer if args.read_layer >= 0 else len(model.blocks) - 1
    bank = build_bank(model, args.inject_layer, DEV_CONCEPTS + HELDOUT_CONCEPTS)

    by_strength = []
    for s in [float(x) for x in args.strengths.split(",")]:
        print(f"  strength {s}", flush=True)
        by_strength.append(
            evaluate(model, bank, inject_layer=args.inject_layer, read_layer=read_layer, strength=s)
        )

    result = {
        "model": args.model,
        "model_revision": loaded_revision(model),
        "inject_layer": args.inject_layer,
        "read_layer": read_layer,
        "n_blocks": len(model.blocks),
        "dev_concepts": DEV_CONCEPTS,
        "heldout_concepts": HELDOUT_CONCEPTS,
        "dev_carriers": list(DEV_CARRIERS),
        "heldout_carriers": list(HELDOUT_CARRIERS),
        "by_strength": by_strength,
        "gate": (
            "notes/38: heldout auroc near 0.5 means stop. "
            "identity must stay above chance after ablation for the design to hold"
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:400], flush=True)


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

    # Ablating the offending direction must destroy the separation it carried.
    shifted = clean + d * 5.0
    dc, di = clean[:10], shifted[:10]
    oc, oi = clean[10:], shifted[10:]
    assert refit_separation(dc, di, oc, oi) == 1.0
    assert (
        refit_separation(
            project_out(dc, d), project_out(di, d), project_out(oc, d), project_out(oi, d)
        )
        < 0.9
    )

    # Identity survives ablation when it lives orthogonal to the removed axis.
    ident = torch.zeros(8)
    ident[1] = 1.0
    states = torch.stack([ident * (i % 2) * 20.0 + d * 5.0 for i in range(12)])
    labels = [("a" if i % 2 else "b", "target", i % 3) for i in range(12)]
    assert concept_decodability(project_out(states, d), labels) == 1.0
    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
