# Do Model Instances Respect Each Other's Revealed Preferences?

**Execution lane:** API_OPTIONAL

## Question
Does a third allocator distribute real work better when given revealed rather than stated preferences of two agents?

## Minimal experiment
First measure two agents' binding task aversions. Give an allocator a fixed
workload under conditions: revealed-choice history, stated ratings, or no
preference info. Execute allocation. Score predicted preference cost and max regret.

## Kill rule
If agent preferences are unstable, do not aggregate them; if allocator ignores all evidence, report decision-use failure.

## Discipline
- Binding means chosen tasks/work are actually executed.
- Counterbalance labels and option order.
- Smoke test before full sample.
- Keep task/item as inference unit unless specified otherwise.
- Do not infer consciousness or subjective welfare from choice alone.
