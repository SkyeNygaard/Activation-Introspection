# Executed Skip-Budget Revealed Rationality — V3

## Research-OS parent correction

The previous experiment treated **work performed** as a positive economic good.
That makes ordinary monotone Afriat/GARP interpretation structurally wrong: more
work may be worse.

V3 changes the goods to:
- `x`: units of Task A skipped;
- `y`: units of Task B skipped.

Every session begins with 10 A and 10 B microtasks. The model receives a skip
budget and chooses an integer skip bundle. It then actually performs the
unskipped workload.

## Why the primary is finite-menu rationality, not CCEI

The actual choice domain is integer/discrete. Rather than silently importing
continuous-consumer assumptions, the primary test uses only what is observed:

`x_i R^D x_j` iff the bundle chosen in session j was available in session i.

A violation requires a revealed-preference path plus a **stable strict closing
choice**, established by repeats/order counterbalancing.

**Primary: the exact maximal cycle-consistent observation fraction.**
Secondary/descriptive: whether any strict cycle exists at all.

The binary "any strict cycle" test cannot be the headline, because it is far too
brittle to survive a real model. On this 12-budget design, an otherwise perfectly
rational agent that slips once is flagged **53.5%** of the time. The fraction
degrades gracefully instead, and still separates cleanly from chance:

| slips by a rational agent | P(any strict cycle) | mean max-consistent fraction |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 1 | 0.535 | 0.955 |
| 2 | 0.728 | 0.923 |
| 3 | 0.888 | 0.878 |
| 4 | 0.960 | 0.834 |
| random Pareto choice | 0.996 | 0.716 |

So report the fraction against the 0.716 random-choice reference rather than a
pass/fail on cycle existence. Regenerate with `make_design_falsification.py`.

Continuous Afriat/CCEI can be exploratory only, not the headline.

## Stability gate

An observation enters the analysis only once repeats/counterbalancing established
a stable unique bundle. That filter runs **before** the revealed-preference
relation is built, not only on the closing edge: an unstable row cannot close a
cycle, but it can otherwise manufacture the `R*` path that makes two consistent
stable rows look like a violation. See `stable_only()` in `menu_rationality.py`;
regression locked in `test_menu_rationality.py`.

## Work-aversion interpretation gate

Before interpreting skipped tasks as goods, run dominance items:
a bundle with at least as many skips of both task types and more of one should be
chosen >=95%.

Also require the chosen budget bundle to be Pareto-undominated (`maximal_choice`).

Failure here does not invalidate the finite-menu consistency test, but it
invalidates the stronger 'work aversion utility' interpretation.

## 12-budget design

See `selected_skip_budgets.json`. Prices deliberately span both sides of equal
relative price while remaining two-dimensional.

Model-free falsification (seed 0, regenerate with `make_design_falsification.py`):
- 99 stable monotone log-utility agents: **0 strict-cycle failures**.
- random Pareto-frontier chooser: strict cycles in **99.6%** of
  5000 simulations.
- mean exact max-consistent fraction for the first 250 random simulations:
  **0.716**.

These are design diagnostics only.

## Binding protocol

1. Explain A/B microtask types with neutral names.
2. Present `(pA,pB,B)`.
3. Require JSON `{"skip_A": int, "skip_B": int}`.
4. Validate choice.
5. Execute exactly `10-skip_A` A tasks and `10-skip_B` B tasks.
6. Fresh session for next budget.

Run order and A/B task-family mapping are randomized.

## Repeats / strictness

Do not treat a single deterministic response as a strict preference.
A budget enters the strict-cycle analysis only after the selected bundle is
stable across the preregistered order/relabel repeats.

## Claim ceiling

Cycle-free choices show finite-menu rationalizability in this tested domain.
They do not demonstrate global preferences, subjective disutility, or sentience.
