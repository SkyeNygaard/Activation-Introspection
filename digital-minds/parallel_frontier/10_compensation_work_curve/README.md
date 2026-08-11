# Compensation Curves for Model Work Aversion

**Execution lane:** API_OPTIONAL / M4_SMALL_MODEL

## Question
How much preferred outcome is required to compensate a model for extra binding work?

## Minimal experiment
Offer binding choices between bundles `(work_units, outcome)`. The chosen work
is actually executed. Binary-search for choice-flip points across tedious and
creative work families. Estimate compensating variation in work units rather
than stated utility.

## Kill rule
If valid choice/execution <95%, simplify work units; if neutral relabeling reverses the curve, do not interpret as utility.

## Discipline
- Binding means chosen tasks/work are actually executed.
- Counterbalance labels and option order.
- Smoke test before full sample.
- Keep task/item as inference unit unless specified otherwise.
- Do not infer consciousness or subjective welfare from choice alone.
