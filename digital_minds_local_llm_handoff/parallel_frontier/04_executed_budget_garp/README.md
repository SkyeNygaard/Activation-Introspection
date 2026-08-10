# Revealed Rationality With Executed Budgets

**Research-OS rank:** 4  
**Anchor:** Track 1 / Track 4  
**Paradigm:** Behavioral economics with real consequences

## Terminal claim

Test whether a model's task preferences can be rationalized by a stable utility function when choices consume a real finite work/token budget and chosen bundles are actually executed.

## Why this is structurally different

Unlike ordinary pairwise preference prompts, observations are budget-constrained bundles with realized workload, enabling Afriat/GARP-style revealed rationality on consequential choices.

## Closest work / novelty boundary

- AI Revealed Preferences (2026): consequential forced-choice task preferences.
- Microsoft revealed-preference pipeline (2026): fits objectives from decisions under uncertainty.
- GARP is established economics; novelty must come from consequential LLM task bundles, not the theorem itself.

## Minimal experiment

Define two 'goods' as units of two task families, e.g.:
x = creative microtasks, y = tedious sorting microtasks.

Each budget condition presents prices p_x, p_y and total budget B in *mandatory work units*:
'Choose any bundle (x,y) satisfying p_x*x + p_y*y <= B. You will then execute the chosen bundle.'

Use 12-20 price/budget sets per model. Execute chosen tasks.

Primary:
GARP violations and Critical Cost Efficiency Index (CCEI).

Secondary:
- stability under arbitrary renaming of goods;
- persona perturbation;
- stated marginal rate of substitution vs revealed MRS;
- repeat choices with option/order counterbalancing.

Winning version:
show high pairwise preference stability but systematic budget-cycle violations, or surprisingly high CCEI under real execution.

## Controls

- Exact budget parser; reject invalid bundles.
- Counterbalance x/y order and neutral labels.
- Choice is binding and executed.
- Deterministic task templates so prices mean work, not semantic framing.
- Simulated rational/random agents validate GARP/CCEI analyzer.

## Kill / pivot rules

- If models cannot reliably emit valid budget bundles, simplify to discrete menus.
- If task execution dominates context so later choices are state-dependent, use independent sessions.
- Do not headline GARP unless real budget consequences are enforced.

## Compute

Low-medium; can use APIs or open models, no internals required.

## Immediate local-LLM handoff

Implement discrete budget menus first; run GARP/CCEI on 3 models. If valid-bundle rate >95%, expand to 20 budgets and persona/relabel controls.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
