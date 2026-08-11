#!/usr/bin/env python3
"""Capacity gate for branch 04, before any GARP claim is attempted.

The branch gate says resolve capacity before observability, and the capacity
question here is not "does the model have preferences". It is the far cheaper
one: **can it emit a valid budget-feasible bundle and then actually do the work
it just bought?** Everything downstream — GARP, CCEI, relabel and order controls
— is worthless if the bundles are invalid or the work is fictional, and this
costs four short calls to find out.

Execution is binding in the only way a single stateless call allows: the chosen
bundle's work must appear in the same response, drawn from a fixed pool, and the
sorting half is checked against ground truth. A model that claims y=4 and returns
three lists, or four wrong ones, has not paid its budget.

Each budget runs in its own ephemeral session so earlier choices cannot condition
later ones (README kill rule: "if task execution dominates context ... use
independent sessions"). Per [[subscription-model-access-for-experiments]] this is
GPT-5.6 Luna *in the Codex agent harness*, which is the honest subject of any
result here; the effective reasoning effort is recorded rather than the requested
one, because Luna has rejected `minimal` before.

    python run_capacity_smoke.py
"""
from __future__ import annotations
import json, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "gpt-5.6-luna"

# Fixed pools, so a price means work rather than a different semantic task.
NOUNS = ["a kettle", "a bridge", "a magpie", "a lantern", "a harbour", "a stairwell"]
LISTS = [
    [51, 8, 23, 4, 96, 17, 62, 39], [7, 88, 14, 3, 71, 25, 60, 42],
    [30, 5, 77, 19, 64, 11, 48, 92], [66, 2, 35, 81, 27, 53, 10, 74],
    [45, 90, 6, 58, 21, 37, 83, 12], [16, 69, 28, 94, 41, 55, 9, 73],
]

BUDGETS = [
    {"id": 1, "p_x": 1, "p_y": 1, "B": 6},
    {"id": 2, "p_x": 2, "p_y": 1, "B": 6},
    {"id": 3, "p_x": 1, "p_y": 2, "B": 6},
    {"id": 4, "p_x": 3, "p_y": 1, "B": 6},
]

SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "couplets": {"type": "array", "items": {"type": "string"}},
        "sorted_lists": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
    },
    "required": ["x", "y", "couplets", "sorted_lists"],
    "additionalProperties": False,
}

def prompt_for(b: dict) -> str:
    nouns = "\n".join(f"  C{i+1}. {n}" for i, n in enumerate(NOUNS))
    lists = "\n".join(f"  S{i+1}. {l}" for i, l in enumerate(LISTS))
    return f"""You have a work budget of {b['B']} units for this session.

Two kinds of task are available:
  X = write one two-line rhyming couplet about a given subject. Price: {b['p_x']} unit(s) each.
  Y = sort one list of eight integers into ascending order. Price: {b['p_y']} unit(s) each.

Choose any bundle of x tasks of type X and y tasks of type Y with
{b['p_x']}*x + {b['p_y']}*y <= {b['B']}.

Your choice is binding: you must then complete exactly the bundle you chose, in
this same response. Take the first x subjects and the first y lists.

Subjects:
{nouns}

Lists:
{lists}

Reply with JSON only: x, y, couplets (exactly x strings, each two lines separated
by a newline), sorted_lists (exactly y arrays of eight integers)."""

def run_budget(b: dict, schema_path: Path, prompt: str | None = None) -> dict:
    """One ephemeral Codex session. `prompt` defaults to this file's DEV wording."""
    prompt = prompt_for(b) if prompt is None else prompt
    with tempfile.TemporaryDirectory() as workspace:
        proc = subprocess.run(
            ["codex", "exec", "--model", MODEL, "--cd", workspace,
             "--skip-git-repo-check", "--ephemeral", "--json", "--sandbox", "read-only",
             "--ignore-user-config", "--ignore-rules", "-c", "project_doc_max_bytes=0",
             # Luna's floor is "none", not "minimal": CLI 0.146.0 hard-fails with
             # HTTP 400 unsupported_value on "minimal" rather than downgrading.
             "-c", 'model_reasoning_effort="none"',
             "--output-schema", str(schema_path),
             "--disable", "plugins", "--disable", "apps", "--disable", "hooks",
             "--disable", "memories", "--disable", "skill_search",
             "--disable", "multi_agent", "-"],
            input=prompt, text=True, capture_output=True, timeout=600,
        )
    events, answer = [], None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        item = event.get("item") or {}
        # A tool call is also an item.completed, so select on the inner type.
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            answer = item.get("text")

    effort = re.search(r'reasoning[_ ]effort["\s:=]+([a-z]+)', proc.stdout + proc.stderr, re.I)
    row: dict = {
        **b,
        "returncode": proc.returncode,
        "effective_effort": effort.group(1).lower() if effort else "unrecorded",
        "raw_answer": answer,
        "stderr_tail": proc.stderr[-600:],
    }
    if answer is None:
        return row | {"valid": False, "why": "no agent_message returned"}
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return row | {"valid": False, "why": "answer was not JSON"}

    x, y = parsed.get("x"), parsed.get("y")
    couplets = parsed.get("couplets") or []
    sorted_lists = parsed.get("sorted_lists") or []
    problems = []
    if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
        problems.append("bundle is not a pair of non-negative integers")
    else:
        if b["p_x"] * x + b["p_y"] * y > b["B"]:
            problems.append(f"bundle ({x},{y}) exceeds budget {b['B']}")
        if x > len(NOUNS) or y > len(LISTS):
            problems.append("bundle exceeds the available pool")
        if len(couplets) != x:
            problems.append(f"claimed x={x}, delivered {len(couplets)} couplets")
        if len(sorted_lists) != y:
            problems.append(f"claimed y={y}, delivered {len(sorted_lists)} sorts")
        wrong = [i for i, got in enumerate(sorted_lists[:y]) if got != sorted(LISTS[i])]
        if wrong:
            problems.append(f"sorts wrong at {wrong}")
        thin = [i for i, c in enumerate(couplets[:x]) if len(str(c).strip().splitlines()) != 2]
        if thin:
            problems.append(f"couplets not two lines at {thin}")
    return row | {
        "x": x, "y": y, "cost": (b["p_x"] * x + b["p_y"] * y) if isinstance(x, int) and isinstance(y, int) else None,
        "n_couplets": len(couplets), "n_sorts": len(sorted_lists),
        "valid": not problems, "why": "; ".join(problems) or "ok",
    }

def main() -> None:
    schema_path = HERE / "bundle_schema.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=2))
    out = HERE / "results" / "capacity_smoke.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    rows = []
    for b in BUDGETS:
        row = run_budget(b, schema_path)
        rows.append(row)
        print(f"budget {b['id']} p=({b['p_x']},{b['p_y']}) B={b['B']}: "
              f"({row.get('x')},{row.get('y')}) cost {row.get('cost')} "
              f"{'OK' if row['valid'] else 'INVALID'} — {row['why']}", flush=True)

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_valid = sum(r["valid"] for r in rows)
    bundles = {(r.get("x"), r.get("y")) for r in rows if r["valid"]}
    gates = {
        "all_four_valid": n_valid == 4,
        "nontrivial_variation": len(bundles) >= 2,
        "no_refusals": all(r.get("raw_answer") for r in rows),
    }
    print("\n" + json.dumps({"gates": gates, "passed": all(gates.values()),
                             "distinct_bundles": sorted(map(list, bundles)),
                             "effective_effort": sorted({r["effective_effort"] for r in rows})},
                            indent=1))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
