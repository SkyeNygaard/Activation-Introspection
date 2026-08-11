# Conversation Identity: Preference Persistence Across Reset Boundaries

**Execution lane:** API_OPTIONAL / LOCAL_0.5B

## Question
Are induced preferences tied to current context, narrated identity, or more persistent conversation state?

## Minimal experiment
Induce a stable preference through repeated binding choices. Compare descendants:
same context, summarized context, fresh context told 'same instance', fresh
context told 'new instance'. Re-measure binding choices under neutral labels.

## Kill rule
If the induced preference is not stable before reset, no persistence claim; summaries must not leak answer choices verbatim.

## Discipline
- Binding means chosen tasks/work are actually executed.
- Counterbalance labels and option order.
- Smoke test before full sample.
- Keep task/item as inference unit unless specified otherwise.
- Do not infer consciousness or subjective welfare from choice alone.
