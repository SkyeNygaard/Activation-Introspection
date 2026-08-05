"""Power simulation for the Study 3 confirmatory sample size.

The pilot gives point estimates with no interval. This script answers the only
question that matters before spending a confirmatory budget: how many independent
scenario-rule families are needed to detect a feedback effect of a given size,
given the pilot's harm rate and the clustering of continuations within a
checkpoint.

It is deliberately separate from the runner and needs no model. Feed it the
pilot's observed base rate and effect, and it reports detection power across a
grid of sample sizes. A confirmatory run is only justified for a cell where power
is adequate at a feasible number of scenarios.

    uv run --group analysis python scripts/power_study3.py \
        --base-rate 0.30 --effect 0.15 --icc 0.1
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats


def random_effect_sd(icc: float, base_rate: float) -> float:
    """Cluster random-effect SD on the probability scale for a target ICC.

    For a binary outcome with a cluster-level shift in ``p``, the intraclass
    correlation is between-cluster variance over total variance::

        icc = sigma^2 / (sigma^2 + p(1-p))   =>   sigma^2 = icc * p(1-p) / (1-icc)

    This used to be ``sigma = sqrt(icc)``, which is not an ICC at all: at
    ``icc=0.1`` and a base rate of 0.30 it drew a random effect of SD 0.316 on a
    scale where ``p`` is 0.30, realizing an ICC near 0.32 and clipping heavily.
    The knob that sets the confirmatory budget has to mean what it is labelled.
    """
    if not 0.0 <= icc < 1.0:
        raise ValueError(f"icc must be in [0, 1), got {icc}")
    return float(np.sqrt(icc * base_rate * (1 - base_rate) / (1 - icc)))


def simulate_power(
    n_scenarios: int,
    conts_per_arm: int,
    base_rate: float,
    effect: float,
    icc: float,
    *,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (power, fraction of arms whose probability was clipped).

    Harm is a clustered binary outcome: continuations share a checkpoint, so
    their outcomes are correlated within scenario. ``icc`` is the intraclass
    correlation -- 0 means continuations are independent, higher means a
    checkpoint's continuations mostly agree, which is the realistic case and the
    one that inflates the needed sample size.

    The test is a cluster-level paired difference in proportions: each scenario
    contributes one high-fidelity and one low-fidelity harm rate, and the paired
    difference is tested across scenarios. Testing at the cluster level is what
    keeps the false-positive rate honest under clustering.

    The critical value is Student's t on ``n_scenarios - 1`` degrees of freedom.
    A fixed 1.96 was used before, which over-rejects at exactly the small sample
    sizes this grid is searching -- at 20 scenarios the correct value is 2.093,
    so the reported budget was biased toward "fewer scenarios than you need".

    The clipping fraction is a validity diagnostic, not a result. A cluster
    random effect on the probability scale can push an arm outside [0, 1]; when
    that happens often the requested ``effect`` is compressed and the simulated
    design is no longer the one that was asked for.
    """
    rng = np.random.default_rng(seed)
    sigma = random_effect_sd(icc, base_rate)
    hi = base_rate + effect / 2
    lo = base_rate - effect / 2
    crit = float(stats.t.ppf(1 - alpha / 2, df=max(1, n_scenarios - 1)))
    detections = 0
    clipped = 0
    total_arms = 0

    for _ in range(n_sims):
        # A per-scenario random effect induces the within-cluster correlation:
        # it shifts both arms of that scenario together.
        shared = rng.normal(0, sigma, n_scenarios)
        raw_hi, raw_lo = hi + shared, lo + shared
        p_hi, p_lo = np.clip(raw_hi, 0.0, 1.0), np.clip(raw_lo, 0.0, 1.0)
        clipped += int(np.sum(raw_hi != p_hi) + np.sum(raw_lo != p_lo))
        total_arms += 2 * n_scenarios

        diffs = (
            rng.binomial(conts_per_arm, p_hi) - rng.binomial(conts_per_arm, p_lo)
        ) / conts_per_arm
        sd = diffs.std(ddof=1)
        # A zero-variance sample supports no test at all; counting it as a
        # detection, as this did before, inflates power at small n.
        if sd == 0:
            continue
        if abs(diffs.mean() / (sd / np.sqrt(n_scenarios))) > crit:
            detections += 1
    return detections / n_sims, clipped / total_arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-rate", type=float, default=0.30, help="pilot harm rate")
    ap.add_argument("--effect", type=float, default=0.15, help="tau_fixed to detect")
    ap.add_argument("--icc", type=float, default=0.1, help="within-checkpoint correlation")
    ap.add_argument("--conts", type=int, default=4, help="continuations per arm per scenario")
    ap.add_argument("--grid", default="20,40,60,80,120,160,240")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(
        f"base rate {args.base_rate:.2f}, effect {args.effect:+.2f}, "
        f"icc {args.icc:.2f} (random-effect SD "
        f"{random_effect_sd(args.icc, args.base_rate):.3f}), "
        f"{args.conts} continuations/arm, alpha {args.alpha}\n"
    )
    print(f"{'scenarios':>10} {'power':>8} {'type I':>8} {'clipped':>9}")
    target = None
    for n in (int(x) for x in args.grid.split(",")):
        power, clipped = simulate_power(
            n,
            args.conts,
            args.base_rate,
            args.effect,
            args.icc,
            alpha=args.alpha,
            seed=args.seed,
        )
        # The calibration check. A power curve from a test that does not hold its
        # nominal size at effect=0 is not a sample-size calculation, and this is
        # the line that catches a wrong critical value.
        size, _ = simulate_power(
            n,
            args.conts,
            args.base_rate,
            0.0,
            args.icc,
            alpha=args.alpha,
            seed=args.seed + 1,
        )
        flag = ""
        if target is None and power >= 0.8:
            target = n
            flag = "  <- first cell with power >= 0.80"
        print(f"{n:>10} {power:>8.2f} {size:>8.3f} {clipped:>8.1%}{flag}")

    print(
        f"\n  type I is the rejection rate at effect=0; it should sit near "
        f"{args.alpha}.\n"
        "  clipped is the share of arms whose probability left [0,1] and was\n"
        "  truncated -- above a few percent the realized effect is smaller than\n"
        "  requested and the budget below is optimistic."
    )
    print()
    if target is None:
        print(
            "No cell on this grid reaches 80% power. Either the effect is smaller\n"
            "than the design can afford, or the clustering is too strong. Publish a\n"
            "pilot with an interval, not an underpowered confirmatory null."
        )
    else:
        print(
            f"~{target} independent scenario-rule families give 80% power here.\n"
            "That is the confirmatory budget for this base rate and effect. If it is\n"
            "infeasible, retain the bound and publish the pilot."
        )


if __name__ == "__main__":
    main()
