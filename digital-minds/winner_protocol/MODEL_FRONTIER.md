# Model Frontier

## Why model choice is a root branch

The experiment needs two capacities at the **same depth**:

1. the reward direction must have a validated functional-welfare effect;
2. a transient hidden-state edit must remain usable/reportable downstream.

Those are not guaranteed to co-occur in one model.

## A — Qwen3-4B-Instruct-2507

**Pros**
- closest to the original functional-welfare paper;
- public vGOLD/vMOLD artifacts;
- 4B is cheap and fast;
- user's prior causal codebook success is in the Qwen family.

**Cons**
- public replication reports only moderate late-third recruitment (~-0.54);
- source welfare steering is late (roughly L17–26), while retained-trace usability
  can decline with depth.

**Initial DEV cell**
- layer 29 (the public fork's validated Qwen welfare/emotion-alignment layer);
- factor 2.

If semantic steering is healthy but codebook reportability is weak, move earlier in one bounded sequence: L26 → L23. Only then consider factor 4.

## B — Llama-3.1-8B-Instruct

**Pros**
- public cross-model replication reports strong recruitment (~-0.86);
- cleaner off-task sentiment steering signature;
- emotion alignment is extremely axis-like in the public replication;
- independent quantitative-introspection work reports very high introspective
  coupling for some measures at Llama-3.1-8B scale.

**Cons**
- opaque codebook method has not yet been demonstrated in this exact model;
- 8B costs more;
- vector comes from an independent replication/extension rather than the
  original paper's primary model.

**Initial DEV cells**
- welfare-relevant neighborhood around layer 20;
- start L18/L20/L22, factor 2; factor 4 only if factor 2 is valid but weak.

## Current allocation

Do **not** spend eight Qwen hyperparameter cells before testing Llama.

Recommended first GPU pass:

| model | layer | factor | purpose |
|---|---:|---:|---|
| Qwen3-4B | 29 | 2 | artifact-specific validated welfare layer |
| Llama-3.1-8B | 20 | 2 | high-signal branch |

For each, run one DEV carrier with exact 24 cells in target + query-only and the
semantic fork.

Promote a model if:
- target-query-only p(correct) >= +0.10;
- semantic +/- state effect has expected sign;
- label mass and model behavior stay healthy.

If exactly one passes, concentrate there.
If both pass, prefer the larger target-query contrast subject to healthy semantic
effect, then use the other as a replication if time.
If neither passes, expand **one bounded neighborhood** around the more promising
model before pruning the structured-report family.

## Phase-2 negative-control model

OLMo-3-7B is scientifically valuable because the public replication reports
weak recruitment and failure of both the steering-X and emotion-line validation.

After a positive recruited-model result:
- run the same structured report on OLMo's reward direction;
- interpret codebook access and semantic welfare coupling separately.

Interesting outcomes:

1. **Opaque access in both; semantic coupling only recruited model**
   - excellent evidence that activation reportability is generic while
     welfare-like meaning requires recruitment.

2. **Both access and semantic coupling track recruitment**
   - suggests accessibility itself changes with learned representational geometry.

3. **Neither OLMo readout works**
   - compatible with either weak direction or weak reporting channel; needs a
     generic hidden-state positive control before interpretation.
