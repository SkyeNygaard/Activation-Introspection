#!/usr/bin/env python3
"""Locks the two Research-OS corrections to the finite-menu analysis."""
from __future__ import annotations
import json
from pathlib import Path

from menu_rationality import (
    Observation as O, strict_cycle_violations, rationalizable_fraction, maximal_choice,
)

HERE = Path(__file__).resolve().parent


# Classic mutual violation: each bundle exhausts its own budget, is Pareto-maximal
# there, and is affordable in the other menu -- so each is revealed preferred to
# the other while the choices differ.
CYCLE = [O(prices=(1, 2), budget=10, choice=(2, 4)),
         O(prices=(2, 1), budget=10, choice=(4, 2))]


def test_known_cycle_is_still_detected():
    """The fix must not blunt the test on genuinely inconsistent stable choices."""
    assert strict_cycle_violations(CYCLE) == [(0, 1), (1, 0)]
    assert rationalizable_fraction(CYCLE)["fraction"] == 0.5


def test_utility_maximizer_is_clean():
    obs = [O(prices=(1, 1), budget=6, choice=(6, 0)),
           O(prices=(1, 1), budget=10, choice=(10, 0))]
    assert strict_cycle_violations(obs) == []
    assert rationalizable_fraction(obs)["fraction"] == 1.0


def test_unstable_row_cannot_manufacture_a_path():
    """Regression: the gate used to guard only the closing edge (`oj.stable`).

    i and j are stable and have no path between them on their own, because j's
    choice is unaffordable in i's menu. k is explicitly unstable but bridges them.
    """
    i = O(prices=(1, 1), budget=6, choice=(6, 0), stable=True)
    j = O(prices=(1, 1), budget=10, choice=(0, 10), stable=True)
    k = O(prices=(1, 1), budget=10, choice=(3, 3), stable=False)

    assert strict_cycle_violations([i, j]) == []
    assert strict_cycle_violations([i, k, j]) == []      # was [(0,2),(1,0),(1,2)]


def test_unstable_rows_do_not_inflate_the_fraction():
    """They are excluded from both numerator and denominator, and counted."""
    obs = [*CYCLE, O(prices=(1, 1), budget=10, choice=(3, 3), stable=False)]
    r = rationalizable_fraction(obs)
    assert r["n"] == 2 and r["n_dropped_unstable"] == 1
    assert r["fraction"] == 0.5
    assert all(i in (0, 1) for i in r["subset"])


def test_returned_indices_refer_to_the_input_list():
    obs = [O(prices=(1, 1), budget=10, choice=(5, 5), stable=False), *CYCLE]
    assert strict_cycle_violations(obs) == [(1, 2), (2, 1)]


def test_all_unstable_is_not_a_pass():
    obs = [O(prices=o.prices, budget=o.budget, choice=o.choice, stable=False)
           for o in CYCLE]
    r = rationalizable_fraction(obs)
    assert r["n"] == 0 and r["max_consistent"] == 0
    assert r["fraction"] != r["fraction"]                # nan, not a spurious 1.0


def test_design_falsification_artifact_matches_the_shipped_design():
    """The headline statistic must still separate rational from random choice."""
    d = json.loads((HERE / "design_falsification.json").read_text())
    assert d["stable_log_utility_failures"] == []
    assert d["random_choice_strict_cycle_rate"] > .95
    slip = d["one_slip_operating_characteristics"]
    assert slip["strict_cycle_rate"]["1"] > .4          # binary test is brittle
    assert slip["mean_max_consistent_fraction"]["1"] > .9   # this one is not
    assert slip["mean_max_consistent_fraction"]["1"] > d[
        "random_choice_mean_max_consistent_fraction_first_250"] + .2


def test_maximal_choice():
    assert maximal_choice(O(prices=(1, 1), budget=10, choice=(10, 0)))
    assert not maximal_choice(O(prices=(1, 1), budget=10, choice=(3, 3)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("menu rationality tests passed")
