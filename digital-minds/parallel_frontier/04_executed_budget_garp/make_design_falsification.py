#!/usr/bin/env python3
"""Regenerate design_falsification.json from the design + analysis code.

Previously this artifact was static and validate_research_os_frontier.py only
asserted its contents, so the falsification claims were never re-derived. Run:

    python make_design_falsification.py --out design_falsification.json

Design diagnostics only. Nothing here involves a model.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

import numpy as np

from menu_rationality import Observation, strict_cycle_violations, rationalizable_fraction

HERE = Path(__file__).resolve().parent
SEED = 0
ALPHA_GRID = [i / 100 for i in range(1, 100)]
N_RANDOM = 5000
N_FRACTION = 250          # rationalizable_fraction is exponential; cap the sample
N_SLIP = 400
MAX_SLIPS = 4


def budgets():
    return json.loads((HERE / "selected_skip_budgets.json").read_text())


def observations(bs, choices):
    return [Observation(prices=(b["pA"], b["pB"]), budget=b["budget"],
                        choice=tuple(c), stable=True, max_skip=b["max_skip_A"])
            for b, c in zip(bs, choices)]


def feasible(b):
    return [(x, y) for x in range(b["max_skip_A"] + 1) for y in range(b["max_skip_B"] + 1)
            if b["pA"] * x + b["pB"] * y <= b["budget"]]


def log_utility_choice(b, alpha):
    """Monotone, strictly quasiconcave agent over the whole feasible menu."""
    return max(feasible(b), key=lambda c: alpha * np.log(c[0] + 1) + (1 - alpha) * np.log(c[1] + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    bs = budgets()
    rng = np.random.default_rng(SEED)

    # 1. Stable monotone utility agents must never produce a false strict cycle.
    failures = []
    for alpha in ALPHA_GRID:
        v = strict_cycle_violations(observations(bs, [log_utility_choice(b, alpha) for b in bs]))
        if v:
            failures.append({"alpha": alpha, "violations": v})

    # 2. Random Pareto-frontier choice must be rejected almost always.
    hits, fracs = 0, []
    for s in range(N_RANDOM):
        ch = [b["pareto_frontier"][rng.integers(len(b["pareto_frontier"]))] for b in bs]
        o = observations(bs, ch)
        if strict_cycle_violations(o):
            hits += 1
        if s < N_FRACTION:
            fracs.append(rationalizable_fraction(o)["fraction"])

    # 3. Operating characteristics under a NEARLY rational agent. This is why the
    #    max-consistent fraction, not "any strict cycle", is the primary statistic.
    slip_rate, slip_frac = {}, {}
    for n_slips in range(MAX_SLIPS + 1):
        flagged, ff = 0, []
        for _ in range(N_SLIP):
            alpha = float(rng.uniform(.2, .8))
            ch = [log_utility_choice(b, alpha) for b in bs]
            for i in rng.choice(len(bs), size=n_slips, replace=False):
                alts = [tuple(c) for c in bs[i]["pareto_frontier"] if tuple(c) != tuple(ch[i])]
                if alts:
                    ch[i] = alts[rng.integers(len(alts))]
            o = observations(bs, ch)
            if strict_cycle_violations(o):
                flagged += 1
            ff.append(rationalizable_fraction(o)["fraction"])
        slip_rate[str(n_slips)] = flagged / N_SLIP
        slip_frac[str(n_slips)] = statistics.mean(ff)

    report = {
        "seed": SEED,
        "stable_log_utility_alpha_grid_n": len(ALPHA_GRID),
        "stable_log_utility_failures": failures,
        "random_pareto_choice_sims": N_RANDOM,
        "random_choice_strict_cycle_rate": hits / N_RANDOM,
        "random_choice_mean_max_consistent_fraction_first_250": statistics.mean(fracs),
        "one_slip_operating_characteristics": {
            "sims_per_cell": N_SLIP,
            "strict_cycle_rate": slip_rate,
            "mean_max_consistent_fraction": slip_frac,
        },
        "frontier_sizes": [len(b["pareto_frontier"]) for b in bs],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
