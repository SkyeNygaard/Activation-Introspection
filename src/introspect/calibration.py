"""Calibrate injection strength to a target behavioural effect, per layer.

Why this exists. Sweeping a fixed alpha across layers confounds two things:

- whether a layer supports introspective access, and
- how much damage the injection does at that layer.

The first sweep made this concrete. At L9 the behavioural KL ran 0.02 -> 2.7
across the strengths probed; at L19 the same strengths gave only 0.03 -> 0.35.
So "L9 at alpha=0.2" and "L19 at alpha=0.2" are not the same experiment, and a
layer profile built from them is largely a damage profile.

The fix is to hold the *effect* constant rather than the cause: for each layer,
solve for the alpha that produces a target KL from clean, then run the design at
those per-layer alphas. Reported results should name the target KL, not the
alpha.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from introspect.concepts import ConceptVector
from introspect.hooks import Intervention, intervene
from introspect.metrics import kl_from_clean
from introspect.models import LoadedModel
from introspect.prompts import NEUTRAL_TASK_VARIANTS


@dataclass(frozen=True)
class Calibration:
    layer: int
    target_kl: float
    strength: float
    achieved_kl: float
    converged: bool

    def __str__(self) -> str:
        flag = "" if self.converged else "  (did not converge)"
        return (
            f"L{self.layer:<3} target KL={self.target_kl:.3f}  "
            f"alpha={self.strength:.4f}  achieved={self.achieved_kl:.3f}{flag}"
        )


@torch.no_grad()
def measure_kl(
    model: LoadedModel,
    vectors: Sequence[ConceptVector],
    layer: int,
    strength: float,
    *,
    n_prompts: int = 3,
) -> float:
    """Mean KL from clean across several concepts and prompts.

    Averaged over concepts because individual directions differ several-fold in
    how much they perturb the output; calibrating on one concept would leave the
    others badly off target.
    """
    kls: list[float] = []
    for text in NEUTRAL_TASK_VARIANTS[:n_prompts]:
        prompt = model.chat(text)
        ids = model.encode(prompt)
        clean = model.forward_logits(ids)
        for vec in vectors:
            iv = Intervention(layer=layer, direction=vec.vector, strength=strength)
            with intervene(model, [iv], prompt_len=int(ids.shape[1])):
                perturbed = model.forward_logits(ids)
            kls.append(kl_from_clean(clean[:, -1], perturbed[:, -1]))
    return sum(kls) / len(kls)


def calibrate_layer(
    model: LoadedModel,
    vectors: Sequence[ConceptVector],
    layer: int,
    target_kl: float,
    *,
    lo: float = 0.001,
    hi: float = 1.0,
    tol: float = 0.05,
    max_iter: int = 12,
) -> Calibration:
    """Bisect on strength to hit ``target_kl``.

    KL is monotone in strength in the usable range, so bisection is sufficient
    and needs no gradients. It is bracketed rather than unbounded because past
    alpha ~1.0 the injection is larger than the residual itself and the model
    stops producing usable text -- a regime where matching KL is meaningless
    because the output is word salad either way.
    """
    kl_lo = measure_kl(model, vectors, layer, lo)
    kl_hi = measure_kl(model, vectors, layer, hi)

    if kl_hi < target_kl:
        # Even maximum usable strength cannot reach the target at this layer.
        return Calibration(layer, target_kl, hi, kl_hi, converged=False)
    if kl_lo > target_kl:
        return Calibration(layer, target_kl, lo, kl_lo, converged=False)

    a, b = lo, hi
    mid, kl_mid = hi, kl_hi
    for _ in range(max_iter):
        mid = (a + b) / 2
        kl_mid = measure_kl(model, vectors, layer, mid)
        if abs(kl_mid - target_kl) / target_kl < tol:
            return Calibration(layer, target_kl, mid, kl_mid, converged=True)
        if kl_mid < target_kl:
            a = mid
        else:
            b = mid
    return Calibration(layer, target_kl, mid, kl_mid, converged=False)


def calibrate(
    model: LoadedModel,
    banks: dict[int, dict[str, ConceptVector]],
    target_kl: float,
    *,
    n_concepts: int = 4,
) -> dict[int, Calibration]:
    """Calibrate every layer in ``banks`` to the same behavioural effect."""
    out: dict[int, Calibration] = {}
    for layer, bank in banks.items():
        vectors = [bank[name] for name in sorted(bank)[:n_concepts]]
        out[layer] = calibrate_layer(model, vectors, layer, target_kl)
    return out
