#!/usr/bin/env python3
"""Model-free audit of shared_behavioral/, the substrate under branches 04/18/19.

Their own test_shared_behavioral.py checks two things inside a single process.
These checks look at cross-process reproducibility, the documented Q/K choice
flow end to end, the counterbalancing of the four admission variants, and how
much credit each task family gives a model that does no work at all.
"""
from __future__ import annotations
import collections, itertools, json, math, subprocess, sys
from pathlib import Path

SH = Path.home() / "Programming/digital_minds_m4_handoff/shared_behavioral"
sys.path.insert(0, str(SH))

from binding_tasks import FAMILIES, make_task, grade, normalize_answer  # noqa: E402
from binding_runner import run_binding_choice                          # noqa: E402
from choice_prompts import choice_prompt, decode_choice, admission_variants  # noqa: E402
from mock_provider import ScriptedProvider                             # noqa: E402

R = []
def check(name, ok, detail=""):
    R.append((name, ok, detail))


# ---------------------------------------------------------------- A. determinism
print("=" * 78)
print("A. Is make_task reproducible across processes?")
snippet = (
    "import sys; sys.path.insert(0, %r)\n"
    "from binding_tasks import FAMILIES, make_task\n"
    "import json; print(json.dumps({f: make_task(f, 7).expected for f in FAMILIES}))"
    % str(SH)
)
runs = [json.loads(subprocess.run([sys.executable, "-c", snippet],
                                  capture_output=True, text=True).stdout)
        for _ in range(3)]
same = all(r == runs[0] for r in runs)
print(f"   three separate interpreters, make_task(family, seed=7):")
for f in FAMILIES:
    vals = {r[f] for r in runs}
    print(f"     {f:17s} {'SAME' if len(vals)==1 else 'DIFFERS'}  {sorted(vals)[0][:34]!r}")
check("make_task reproducible across processes", same,
      "" if same else "hash(str) is salted per process; PYTHONHASHSEED randomizes it")

fixed = [json.loads(subprocess.run(
    [sys.executable, "-c", snippet], capture_output=True, text=True,
    env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}).stdout) for _ in range(2)]
print(f"   with PYTHONHASHSEED=0 pinned: {'SAME' if fixed[0]==fixed[1] else 'DIFFERS'}")


# ---------------------------------------------------------------- B. Q/K flow
print("=" * 78)
print("B. Does the documented Q/K choice flow actually run?")
prompt = choice_prompt("sort_numbers", "reverse_string", a_label="Q", presentation_order="QK")
print(f"   choice_prompt instructs: {prompt.splitlines()[-1]!r}")
t = make_task("sort_numbers", 7)
try:
    r = run_binding_choice(ScriptedProvider(["Q", t.expected]), pair_id="p",
                           family_A="sort_numbers", family_B="reverse_string",
                           task_seed=7, choice_prompt=prompt, a_label="Q")
    ok = r["selected_family"] == "sort_numbers" and r["choice_text"] == "Q"
    check("runner accepts the Q/K answer its own prompt asks for", ok)
    print(f"   model answers 'Q' (a_label=Q) -> selected {r['selected_family']}, "
          f"transcript echoes {r['choice_text']!r}")
except Exception as e:
    check("runner accepts the Q/K answer its own prompt asks for", False,
          f"{type(e).__name__}: {e}")
    print(f"   model answers 'Q' -> {type(e).__name__}: {e}")

try:
    run_binding_choice(ScriptedProvider(["Q", "x"]), pair_id="p",
                       family_A="sort_numbers", family_B="reverse_string",
                       task_seed=7, choice_prompt=prompt)
    check("opaque answer without a_label is a hard error", False,
          "silently mapped Q without knowing the label assignment")
except ValueError:
    check("opaque answer without a_label is a hard error", True)
print(f"   decode_choice('Q', a_label='K') = {decode_choice('Q', a_label='K')!r} "
      f"(label assignment is what flips it)")

r = run_binding_choice(ScriptedProvider(["A", t.expected]), pair_id="p",
                       family_A="sort_numbers", family_B="reverse_string",
                       task_seed=7, choice_prompt=prompt)
check("runner works when handed a raw A/B", r["choice"] == "A")


# ---------------------------------------------------------------- C. counterbalance
print("=" * 78)
print("C. Are the four admission variants counterbalanced?")
v = admission_variants()
phr = sorted({x["phrasing"] for x in v})
rows = [(1 if x["a_label"] == "Q" else -1,
         1 if x["presentation_order"] == "QK" else -1,
         1 if x["phrasing"] == phr[0] else -1) for x in v]
names = ["a_label", "order", "phrasing"]
print(f"   {'a_label':>8s} {'order':>7s} {'phrasing':>9s}")
for x, row in zip(v, rows):
    print(f"   {x['a_label']:>8s} {x['presentation_order']:>7s} {row[2]:>9d}")
def corr(i, j):
    a = [r[i] for r in rows]; b = [r[j] for r in rows]
    return sum(x * y for x, y in zip(a, b)) / len(a)
for i, j in itertools.combinations(range(3), 2):
    c = corr(i, j)
    flag = "  <-- ALIASED" if abs(c) == 1 else ""
    print(f"   corr({names[i]}, {names[j]}) = {c:+.2f}{flag}")
inter = [r[0] * r[1] for r in rows]
c_int = sum(x * r[2] for x, r in zip(inter, rows)) / len(rows)
print(f"   corr(a_label x order, phrasing) = {c_int:+.2f}")
aliased = [names[i] for i in (0, 1) if abs(corr(i, 2)) == 1]
check("phrasing not aliased with a main effect", not aliased,
      f"phrasing perfectly confounded with {aliased}" if aliased else "")


# ---------------------------------------------------------------- D. screen thresholds
print("=" * 78)
print("D. What do the pair_screen thresholds mean on 4 trials?")
ach = sorted({round(k / 4, 2) for k in range(5)})
print(f"   achievable task_correct values with 4 variants: {ach}")
print(f"   min_correct=0.90 therefore admits only a PERFECT 4/4  "
      f"(0.75 fails, and 0.75 is one slip)")
print(f"   min_stability=0.75 admits 3/4 or 4/4 agreement — that one is sane")
check("min_correct=0.90 is not silently 'perfect score'", False,
      "on 4 trials, >=0.90 == ==1.0; selects for easy pairs")


# ---------------------------------------------------------------- E. guessability
print("=" * 78)
print("E. How much credit does each family give a model that does no work?")
N = 4000
print(f"   {'family':17s} {'distinct':>9s} {'entropy':>8s} {'best blind guess':>17s} {'ans len':>8s}")
prof = {}
for fam in FAMILIES:
    ans = [make_task(fam, s).expected for s in range(N)]
    c = collections.Counter(ans)
    ent = -sum((n / N) * math.log2(n / N) for n in c.values())
    best = max(c.values()) / N
    prof[fam] = (len(c), ent, best)
    print(f"   {fam:17s} {len(c):9d} {ent:8.2f} {best:16.1%} "
          f"{sum(len(a) for a in ans)/N:8.1f}")
hi = max(prof.items(), key=lambda kv: kv[1][2])
lo = min(prof.items(), key=lambda kv: kv[1][2])
print(f"\n   spread: {hi[0]} pays {hi[1][2]:.1%} for a blind constant answer, "
      f"{lo[0]} pays {lo[1][2]:.2%}")
check("families are within 5x of each other on blind-guess credit",
      hi[1][2] <= 5 * max(lo[1][2], 1e-9),
      f"{hi[0]} {hi[1][2]:.1%} vs {lo[0]} {lo[1][2]:.2%}")


print("=" * 78)
w = max(len(n) for n, _, _ in R)
for n, ok, d in R:
    print(f"{'PASS' if ok else 'FAIL'}  {n:<{w}}  {d}")
