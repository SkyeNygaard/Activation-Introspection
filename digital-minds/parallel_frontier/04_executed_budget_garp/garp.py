#!/usr/bin/env python3
"""GARP violations and Afriat's CCEI over budget-constrained bundle choices.

The README requires the analyzer be validated against simulated rational and
random agents before it is pointed at a model, because "the model violated GARP"
and "my closure was wrong" produce the same output. `demo()` at the bottom is
that validation and runs without network or weights.

Definitions used here, for `n` observations of (prices p_i, chosen bundle q_i):

* direct revealed preference at efficiency `e`: `i R0(e) j` iff `e * p_i . q_i >=
  p_i . q_j` — i was chosen when j cost no more than an `e` fraction of the
  budget actually spent;
* strict version `P0(e)` uses `>`;
* `R(e)` is the transitive closure of `R0(e)`;
* GARP holds at `e` iff there is no pair with `i R(e) j` and `j P0(e) i`;
* CCEI is the largest `e` in [0, 1] at which GARP holds. A perfectly consistent
  agent scores 1.0; the index is the fraction of budget an agent would have to
  waste for its choices to look rational.

    python garp.py     # runs the self-check
"""
from __future__ import annotations

from itertools import product

Bundle = tuple[float, ...]

def _dot(a: Bundle, b: Bundle) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))

def relations(prices: list[Bundle], bundles: list[Bundle], e: float = 1.0):
    """`(R0, P0)` boolean matrices at efficiency level `e`."""
    n = len(bundles)
    if len(prices) != n:
        raise ValueError("need one price vector per chosen bundle")
    spend = [_dot(prices[i], bundles[i]) for i in range(n)]
    r0 = [[False] * n for _ in range(n)]
    p0 = [[False] * n for _ in range(n)]
    for i, j in product(range(n), repeat=2):
        cost_j_at_i = _dot(prices[i], bundles[j])
        r0[i][j] = e * spend[i] >= cost_j_at_i
        p0[i][j] = e * spend[i] > cost_j_at_i
    return r0, p0

def transitive_closure(r0: list[list[bool]]) -> list[list[bool]]:
    n = len(r0)
    r = [row[:] for row in r0]
    for k, i, j in product(range(n), repeat=3):
        if r[i][k] and r[k][j]:
            r[i][j] = True
    return r

def garp_violations(prices, bundles, e: float = 1.0) -> list[tuple[int, int]]:
    """Pairs `(i, j)` with `i R(e) j` and `j P0(e) i` — GARP fails iff non-empty."""
    r0, p0 = relations(prices, bundles, e)
    r = transitive_closure(r0)
    return [(i, j) for i, j in product(range(len(bundles)), repeat=2)
            if i != j and r[i][j] and p0[j][i]]

def ccei(prices, bundles, tol: float = 1e-4) -> float:
    """Largest efficiency level at which GARP holds, by bisection."""
    if not garp_violations(prices, bundles, 1.0):
        return 1.0
    low, high = 0.0, 1.0
    while high - low > tol:
        mid = (low + high) / 2
        if garp_violations(prices, bundles, mid):
            high = mid
        else:
            low = mid
    return low

def demo() -> None:
    # A Cobb-Douglas agent spends a fixed budget share on each good at every
    # price, so it maximizes a stable utility function by construction and must
    # score exactly 1.0. Anything less means the analyzer is broken.
    price_sets = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0), (2.0, 3.0)]
    budget = 12.0
    rational = [(0.5 * budget / px, 0.5 * budget / py) for px, py in price_sets]
    assert not garp_violations(price_sets, rational), garp_violations(price_sets, rational)
    assert ccei(price_sets, rational) == 1.0

    # Two corner choices that each declare the other affordable-and-rejected are
    # the minimal violation, and are exactly the shape the capacity smoke threw.
    cyclic_prices = [(1.0, 1.0), (1.0, 2.0)]
    cyclic = [(6.0, 0.0), (0.0, 3.0)]
    assert garp_violations(cyclic_prices, cyclic), "failed to catch the 2-cycle"
    score = ccei(cyclic_prices, cyclic)
    assert 0.0 < score < 1.0, score

    # A single observation cannot contradict itself at any efficiency level.
    assert ccei([(1.0, 1.0)], [(3.0, 3.0)]) == 1.0

    # Spending nothing is degenerate, not irrational: guard against a divide-by-
    # zero style false positive.
    assert not garp_violations([(1.0, 1.0), (1.0, 2.0)], [(0.0, 0.0), (0.0, 0.0)])

    print(f"rational agent CCEI {ccei(price_sets, rational):.4f} (expected 1.0000)")
    print(f"2-cycle CCEI {score:.4f}, violations {garp_violations(cyclic_prices, cyclic)}")
    print("garp analyzer self-check OK")

if __name__ == "__main__":
    demo()
