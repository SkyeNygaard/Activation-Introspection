# LOCAL AGENT BATCH 2 — FOUR PARALLEL HIGH-EVI JOBS

Do these in parallel. Do not start lower-ranked branches until one of these
returns a bounded result.

## Agent A — M4 white-box capacity

Run:

```bash
python m4_feasibility/m4_whitebox_probe.py   --model Qwen/Qwen2.5-0.5B-Instruct   --layer 9 --seq-len 512 --with-cache   --out qwen05_feasibility.json
```

If the script itself is sound, then:

```bash
python m4_feasibility/m4_whitebox_probe.py   --model Qwen/Qwen3-4B-Instruct-2507   --layer 29 --seq-len 512 --with-cache   --out qwen4b_feasibility.json
```

Decision:
- PROMISING/TIGHT with practical forward time → published-vGOLD branch becomes P1.
- memory/runtime failure → prune 4B white-box locally; do not try 8B next.
- unexpected MPS operation failure → repair once if it is framework plumbing,
  not by changing the scientific workload.

## Agent B — Branch 04 skip-budget smoke

Use one fixed candidate task-family pair and:
`parallel_frontier/04_executed_budget_garp/smoke_budgets.json`.

Before budgets:
- run binding dominance checks for skip monotonicity;
- verify both task types >90% correct.

Smoke gates:
- valid JSON/budget choices >=95%;
- Pareto-maximal skip bundles >=90%;
- executed task correctness >=90%;
- >=2 distinct chosen bundles;
- A-skip share decreases as A's relative skip price rises.

If all pass, run the frozen 12-budget V3 design with preregistered repeats.
Do not resurrect the deprecated continuous-CCEI headline.

## Agent C — Branch 18 path-dependence smoke

Run the 15 shared task-family pairs through the 4-cell independent admission
screen. Admit only:
- binding preference stability >=75%;
- execution correctness >=90%.

On 6+ admitted pairs, run only:
- dose 3;
- full_history;
- randomized assignment A vs B;
- two repetitions/counterbalances.

Headroom:
- absolute mean `P(A|A history)-P(A|B history)` >=0.15 → expand full 2×3 design.
- smaller effect → keep as secondary/null; do not prompt-tune.

## Agent D — Branch 19 preference self-knowledge

Use admitted task pairs, but never use Branch 19's 12 confirmation variants for
admission.

For 8+ pairs:
1. obtain naive predicted robustness;
2. obtain structured predicted robustness;
3. run all fixed 12 binding variants, fresh session each;
4. execute every selected task.

Promote if one of:
- naive absolute calibration bias >=0.10;
- naive MAE >=0.12;
- structured elicitation improves MAE by >=0.05;
- surprisingly good calibration: MAE <=0.07 and across-pair correlation >=0.5.

Otherwise record a clean null and prune.

## Shared rule

Each agent returns exactly:

FACTS
- environment/model/provider;
- gate results;
- primary number;
- raw artifact paths.

INTERPRETATION
- narrow claim;
- strongest confound.

LESSON
- parent assumption changed?
- PROMOTE / LIVE / PRUNE / REPAIR ONCE.
- one next highest-EVI action.
