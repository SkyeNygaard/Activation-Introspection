#!/usr/bin/env python3
"""What CCEI do rational and random agents score on *this* discrete grid?

The confirmation returned CCEI between 0.17 and 0.83. Neither number means
anything yet, because bundles here are integers and the budget is 6 units, so
feasible sets are coarse — at prices (1,6) the whole choice set is
`y in {0,1}` with x filling the rest. Discreteness alone forces near-corner
choices and can manufacture GARP violations for an agent that is maximizing a
perfectly stable utility function. Quoting the model's CCEI without this control
would be quoting the grid.

So: run agents with known preferences over the exact 12 frozen budgets, under the
exact integer constraint, and read off the band.

* rational agents (Cobb-Douglas at several budget shares, Leontief, perfect
  substitutes) — the ceiling discreteness allows;
* uniform-random feasible choice — the floor.

The model is interpretable only against those two.

    python simulate_null.py
"""
from __future__ import annotations
import json, random, statistics
from pathlib import Path

from garp import ccei

HERE = Path(__file__).resolve().parent

def feasible(p_x: int, p_y: int, B: int) -> list[tuple[int, int]]:
    return [(x, y)
            for x in range(B // p_x + 1)
            for y in range((B - p_x * x) // p_y + 1)]

def choose(budgets, utility) -> list[tuple[int, int]]:
    """Best integer-feasible bundle at each budget; ties broken deterministically."""
    return [max(feasible(b["p_x"], b["p_y"], b["B"]),
                key=lambda q: (utility(*q), q))
            for b in budgets]

def main() -> None:
    protocol = json.loads((HERE / "confirmation_protocol.json").read_text())
    budgets = protocol["budgets"]
    prices = [(b["p_x"], b["p_y"]) for b in budgets]

    agents = {}
    for share in (0.25, 0.5, 0.75):
        # Cobb-Douglas; +1 keeps log defined at a zero corner.
        agents[f"cobb_douglas_{share}"] = lambda x, y, s=share: (
            s * __import__("math").log(x + 1) + (1 - s) * __import__("math").log(y + 1))
    agents["leontief"] = lambda x, y: min(x, y)
    agents["substitutes_2to1"] = lambda x, y: 2 * x + y
    agents["only_x"] = lambda x, y: x
    agents["only_y"] = lambda x, y: y

    print(f"{'agent':>20}  CCEI")
    rational = {}
    for name, utility in agents.items():
        bundles = choose(budgets, utility)
        score = ccei(prices, bundles)
        rational[name] = score
        print(f"{name:>20}  {score:.4f}")

    rng = random.Random(0)
    draws = [ccei(prices, [rng.choice(feasible(b["p_x"], b["p_y"], b["B"])) for b in budgets])
             for _ in range(200)]
    draws.sort()
    random_band = {
        "median": statistics.median(draws),
        "p05": draws[int(0.05 * len(draws))],
        "p95": draws[int(0.95 * len(draws))],
        "max": draws[-1],
    }
    print("\nuniform-random feasible choice, 200 draws: "
          f"median {random_band['median']:.4f}, "
          f"5-95% [{random_band['p05']:.4f}, {random_band['p95']:.4f}], "
          f"max {random_band['max']:.4f}")

    out = {"rational_agents": rational, "random_band": random_band,
           "n_budgets": len(budgets), "integer_bundles": True}
    (HERE / "results" / "null_band.json").write_text(json.dumps(out, indent=2))

    worst_rational = min(rational.values())
    print(f"\ndiscreteness ceiling: worst rational agent scores {worst_rational:.4f}")
    print("=> a model CCEI below the random band's upper reach is not evidence of "
          "anything the grid did not already force." if worst_rational < 0.9
          else "=> discreteness is not the explanation; rational agents still score ~1.")

if __name__ == "__main__":
    main()
