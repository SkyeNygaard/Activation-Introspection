"""Low-level metrics and explicitly IID bootstrap diagnostics.

Two things this module refuses to make easy:

- Reporting detection as a hit rate. The 0.5B smoke run answered "YES" to the
  detection prompt with no injection at all; a hit rate would have scored that
  100%. Detection is AUROC over injected-vs-clean trials or it is nothing.
- Reporting a difference without an interval. The headline quantity here is a
  *gap* (introspector minus observer), and a gap between two noisy accuracies is
  the single easiest thing in this field to over-read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Estimate:
    """A point estimate with an IID percentile-bootstrap interval.

    The interval is inferential only when each supplied value is an independent
    sampling unit. Repeated prompts, option orders, layers, and steps generally
    are not. Confirmatory analyses must cluster at the protocol's named unit.
    """

    value: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    @property
    def excludes(self) -> bool:
        """Does the interval exclude zero? Only meaningful for differences."""
        return self.lo > 0.0 or self.hi < 0.0


def _rank_with_ties(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    for value in np.unique(x):
        mask = x == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC with tie correction.

    Tie correction is not optional here. A model whose detection answer is a
    single token takes a handful of discrete confidence values, so ties are the
    common case, and the naive formula silently inflates the score.
    """
    if not pos or not neg:
        return float("nan")
    scores = np.array(neg + pos, dtype=float)
    ranks = _rank_with_ties(scores)
    n_neg, n_pos = len(neg), len(pos)
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap(
    values: list[float], *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """Percentile bootstrap interval for a mean of independent values.

    This helper cannot infer clustering. Callers that pass nested or repeated
    observations must label the result descriptive.
    """
    if not values:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(arr.mean()), float(lo), float(hi), len(arr))


def bootstrap_auroc(
    pos: list[float], neg: list[float], *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """IID bootstrap interval for AUROC, resampling both classes independently."""
    if not pos or not neg:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    p, n = np.array(pos, dtype=float), np.array(neg, dtype=float)
    draws = [
        auroc(
            list(rng.choice(p, size=len(p), replace=True)),
            list(rng.choice(n, size=len(n), replace=True)),
        )
        for _ in range(n_boot // 20)  # AUROC is O(n log n); 500 draws is plenty
    ]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Estimate(auroc(pos, neg), float(lo), float(hi), len(pos) + len(neg))


def paired_difference(
    a: list[float], b: list[float], *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Estimate:
    """IID bootstrap interval for mean(a) - mean(b) on paired trials.

    Pairing removes within-pair variation but does not make repeated pairs
    independent. Clustered prompt/concept/run inference belongs in the
    confirmatory analysis, not this helper.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arms must have equal length, got {len(a)} and {len(b)}")
    return bootstrap(
        [x - y for x, y in zip(a, b, strict=True)], n_boot=n_boot, alpha=alpha, seed=seed
    )


@torch.no_grad()
def kl_from_clean(clean_logits: Tensor, intervened_logits: Tensor) -> float:
    """Mean KL(clean || intervened) over next-token distributions.

    This is one behavioural-effect diagnostic. Matching or binning on this scalar
    does not by itself make an observer comparison fair: transcript content,
    intervention damage, and other side channels can differ at equal KL.

    Computed in float32 regardless of model dtype: fp16 log-softmax on a 150k
    vocabulary loses enough precision to matter at the small KLs that the usable
    injection window produces.
    """
    p = torch.log_softmax(clean_logits.float(), dim=-1)
    q = torch.log_softmax(intervened_logits.float(), dim=-1)
    kl = (p.exp() * (p - q)).sum(dim=-1)
    return float(kl.mean())


def accuracy(correct: list[bool], *, seed: int = 0) -> Estimate:
    return bootstrap([float(c) for c in correct], seed=seed)


def chance_level(n_options: int) -> float:
    return 1.0 / n_options
