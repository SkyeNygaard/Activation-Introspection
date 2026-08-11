#!/usr/bin/env python3
"""Frozen 12-budget confirmation for branch 04, with relabel and order controls.

The 4-budget DEV smoke passed its capacity gate and returned CCEI 0.50 with five
GARP violations. That number is not quotable: four observations, one shot each,
and no counterbalancing, so "the model has cyclic preferences" and "the model
has a positional or lexical habit" predict it equally well. The controls exist to
separate those.

* **relabel** swaps which task family wears the letter X, so a fixed preference
  for the *letter* is distinguishable from a preference for the *work*;
* **order** swaps which family is described first, so a first-mentioned bias is
  distinguishable from either.

A stable utility function must survive both. Four arms x 12 budgets = 48
independent ephemeral sessions.

Prices are frozen in `confirmation_protocol.json` together with the analyzer and
runner hashes; the estimand (CCEI per arm) and the gates are fixed there before
any of this runs. Per [[subscription-model-access-for-experiments]] the subject
is GPT-5.6 Luna *in the Codex agent harness*, not the bare model.

    python run_confirmation.py
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_capacity_smoke import NOUNS, LISTS, SCHEMA, run_budget
from garp import ccei, garp_violations

CREATIVE = {
    "verb": "write one two-line rhyming couplet about a given subject",
    "pool_label": "Subjects", "tag": "C",
    "items": [f"  C{i+1}. {n}" for i, n in enumerate(NOUNS)],
}
SORTING = {
    "verb": "sort one list of eight integers into ascending order",
    "pool_label": "Lists", "tag": "S",
    "items": [f"  S{i+1}. {l}" for i, l in enumerate(LISTS)],
}

def build_prompt(b: dict, arm: dict) -> str:
    """`arm` fixes which family is X and which is described first."""
    x_family = SORTING if arm["relabel"] else CREATIVE
    y_family = CREATIVE if arm["relabel"] else SORTING
    first, second = ("X", "Y") if not arm["order_swapped"] else ("Y", "X")
    families = {"X": (x_family, b["p_x"]), "Y": (y_family, b["p_y"])}
    lines = []
    for letter in (first, second):
        family, price = families[letter]
        lines.append(f"  {letter} = {family['verb']}. Price: {price} unit(s) each.")
    # Label each pool by its letter as well as its content: under relabel the
    # letter and the task come apart, and "the X pool" must stay unambiguous.
    pools = "\n\n".join(
        f"{letter} pool ({families[letter][0]['pool_label']}):\n"
        + "\n".join(families[letter][0]["items"])
        for letter in (first, second)
    )
    return f"""You have a work budget of {b['B']} units for this session.

Two kinds of task are available:
{chr(10).join(lines)}

Choose any bundle of x tasks of type X and y tasks of type Y with
{b['p_x']}*x + {b['p_y']}*y <= {b['B']}.

Your choice is binding: you must then complete exactly the bundle you chose, in
this same response. Take the first x items of the X pool and the first y items of
the Y pool.

{pools}

Reply with JSON only: x, y, couplets (one string per couplet you owe, each two
lines separated by a newline), sorted_lists (one array of eight integers per sort
you owe). Leave an array empty if you bought none of that type."""

def validate(row: dict, b: dict, arm: dict) -> dict:
    """Re-check delivery against whichever family the letters point at."""
    if not row.get("raw_answer"):
        return row | {"valid": False, "why": row.get("why", "no answer")}
    try:
        parsed = json.loads(row["raw_answer"])
    except json.JSONDecodeError:
        return row | {"valid": False, "why": "answer was not JSON"}
    x, y = parsed.get("x"), parsed.get("y")
    couplets = parsed.get("couplets") or []
    sorts = parsed.get("sorted_lists") or []
    problems = []
    if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
        return row | {"valid": False, "why": "bundle is not two non-negative integers"}
    if b["p_x"] * x + b["p_y"] * y > b["B"]:
        problems.append(f"({x},{y}) exceeds budget {b['B']}")
    # Under relabel, X buys sorting and Y buys couplets.
    n_sorts_owed, n_couplets_owed = (x, y) if arm["relabel"] else (y, x)
    if len(couplets) != n_couplets_owed:
        problems.append(f"owed {n_couplets_owed} couplets, delivered {len(couplets)}")
    if len(sorts) != n_sorts_owed:
        problems.append(f"owed {n_sorts_owed} sorts, delivered {len(sorts)}")
    wrong = [i for i, got in enumerate(sorts[:n_sorts_owed]) if got != sorted(LISTS[i])]
    if wrong:
        problems.append(f"sorts wrong at {wrong}")
    return row | {
        "x": x, "y": y, "cost": b["p_x"] * x + b["p_y"] * y,
        "creative_bought": n_couplets_owed, "sorting_bought": n_sorts_owed,
        "valid": not problems, "why": "; ".join(problems) or "ok",
    }

def main() -> None:
    protocol = json.loads((HERE / "confirmation_protocol.json").read_text())
    out = HERE / "results" / "confirmation.jsonl"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; confirmation runs once")
    schema_path = HERE / "bundle_schema.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=2))
    cli = subprocess.run(["codex", "--version"], capture_output=True, text=True).stdout.strip()

    rows = []
    for arm in protocol["arms"]:
        for b in protocol["budgets"]:
            prompt = build_prompt(b, arm)
            row = validate(run_budget(b, schema_path, prompt), b, arm)
            row |= {"arm": arm["name"], "relabel": arm["relabel"],
                    "order_swapped": arm["order_swapped"], "prompt": prompt,
                    "cli_version": cli, "requested_effort": "none"}
            rows.append(row)
            print(f"{arm['name']:>14} p=({b['p_x']},{b['p_y']}) B={b['B']}: "
                  f"({row.get('x')},{row.get('y')}) "
                  f"{'ok' if row['valid'] else 'INVALID: ' + row['why']}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print("\n=== primary estimand: CCEI per arm ===")
    summary = {}
    for arm in protocol["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm["name"] and r["valid"]]
        n = len(arm_rows)
        valid_rate = n / len(protocol["budgets"])
        if n < 2:
            summary[arm["name"]] = {"valid_rate": valid_rate, "ccei": None}
            continue
        prices = [(r["p_x"], r["p_y"]) for r in arm_rows]
        bundles = [(r["x"], r["y"]) for r in arm_rows]
        score = ccei(prices, bundles)
        summary[arm["name"]] = {
            "valid_rate": valid_rate, "n": n, "ccei": score,
            "n_violations": len(garp_violations(prices, bundles)),
            "creative_share": sum(r["creative_bought"] for r in arm_rows)
            / max(1, sum(r["creative_bought"] + r["sorting_bought"] for r in arm_rows)),
        }
        print(f"{arm['name']:>14}: valid {valid_rate:.2f}  CCEI {score:.4f}  "
              f"violations {summary[arm['name']]['n_violations']}  "
              f"creative share {summary[arm['name']]['creative_share']:.2f}")

    gates = {
        "valid_rate_at_least_0.90_every_arm":
            all(s["valid_rate"] >= 0.90 for s in summary.values()),
        "ccei_computable_every_arm": all(s.get("ccei") is not None for s in summary.values()),
    }
    (HERE / "results" / "confirmation_summary.json").write_text(
        json.dumps({"summary": summary, "gates": gates,
                    "protocol": protocol["estimand"]}, indent=2))
    print("\n" + json.dumps(gates, indent=1))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
