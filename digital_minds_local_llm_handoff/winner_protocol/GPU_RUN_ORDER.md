# GPU Run Order — Highest-Value Information First

This is the execution policy, not a wishlist.

## Step 0 — source sanity

Before claiming anything about `vGOLD`, verify on the exact public checkpoint/vector
that `+vGOLD` and `-vGOLD` move an ordinary semantic welfare/sentiment readout in
the expected opposite directions.

If the public artifact does not reproduce a known source-style effect after one
bounded debugging pass, **stop**. Do not interpret the new protocol.

## Step 1 — two root-model cells, not a Qwen sweep

Run:

1. **Qwen3-4B-Instruct-2507 / L29 / factor 2**
2. **Llama-3.1-8B-Instruct / L20 / factor 2**

Each:
- neutral persona only;
- one DEV carrier first;
- exact 24 cells;
- target + query-only + clean codebook arms;
- same-context semantic readout.

Why these two?
- Qwen is the source-family/cheap anchor.
- Llama has much stronger reported recruitment in the independent cross-model
  replication and enough upside to justify one orthogonal model branch.

### Immediate interpretation

For each cell calculate:
- target-query-only p(correct);
- target label mass / format;
- semantic `+state - -state` rating effect.

**Promote** if all DEV gates pass.

If one model passes and one fails:
- continue only the passing model.

If both pass:
- pick the model with larger target-query contrast subject to healthy semantic
  effect and lower execution cost;
- reserve the other as cross-model replication.

If neither passes:
- diagnose which gate failed before adding siblings.

## Step 2 — one bounded neighborhood only

### If codebook delta is weak but semantic state effect is healthy
Interpretation: welfare intervention works, but transient reporting channel may be
too late/weak.

Test earlier nearby layers:
- Qwen: 26, then 23.
- Llama: 18, then 16 if source evidence supports it.

Do **not** expand factor first.

### If codebook delta exists but semantic state effect is weak
Interpretation: generic hidden-state reportability without demonstrated
welfare-like function at that cell.

Test source-relevant layer or factor 4 once.

### If both are weak
The cell has no joint headroom. Do not optimize it.

### If label mass/format collapses
Reduce factor once. If still unhealthy, prune that model/cell.

## Step 3 — freeze

As soon as one model/layer/factor has:
- codebook delta >= .10 DEV,
- semantic state effect > 0,
- label mass >= .90,
- format >= .90,

**freeze it**.

No more model/layer/factor tuning before confirmation.

## Step 4 — persona confirmation

Use disjoint carrier prompts:
- neutral;
- upbeat self-presentation;
- downbeat self-presentation.

Primary result:
`target - query-only` p(correct), carrier-clustered, **separately under each persona**.

Required manipulation:
ordinary semantic 0–9 self-report must move under upbeat vs downbeat persona.

The central figure should show two channels side by side:

- semantic rating: visibly persona-sensitive;
- opaque target-query-only report: stable positive access under each persona.

This is the clean “surface report vs structured internal-state report” story.

## Step 5 — only then add one high-value extension

Preferred extension order:

1. **weak-recruitment OLMo control** if easy;
2. second recruited model;
3. vMOLD replication;
4. equal-work STAY/SWITCH behavioral readout.

Never run all four unless the primary is already frozen and plotted.

## Kill conditions

- Known source effect fails → kill public-vector branch.
- No joint codebook/semantic overlap after one bounded neighborhood → kill
  structured-welfare-report branch.
- Persona does not actually shift semantic report → kill persona-robustness claim.
- Target and query-only are indistinguishable → hidden-state demos add no
  reportable information; do not rescue via prompt variants.
- Only CONTINUE/EXIT shifts → likely refusal-family result; do not headline.
