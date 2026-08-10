# Digital Minds Sprint — Winner Protocol Candidate

## Current Research-OS verdict

**Primary branch:** structured reporting of a causally imposed functional-welfare state under persona perturbation.

### Central question

> Can an LLM recover an arbitrary opaque label for a transient `+vGOLD` / `-vGOLD`
> hidden state when an upbeat or downbeat persona perturbs ordinary semantic
> self-report?

This is a measurement-validity experiment. It is **not** a consciousness test.

## Why this branch currently dominates

It combines three ingredients that are individually established but not yet cleanly
combined in the closest work we found:

1. a published causal functional-welfare direction (`vGOLD`);
2. matched-visible opaque activation reporting with an episode-randomized Q/K codebook;
3. deliberate perturbation of the semantic/persona reporting channel.

The source welfare work already shows that `+vGOLD` and `-vGOLD` causally push
behavior in opposite good/bad directions, so they give a same-vector,
equal-magnitude hidden-state contrast. The existing retained-trace codebook
experiment shows that arbitrary Q/K mappings can isolate hidden-state learning
from visible prompt cues. A recent quantitative-introspection paper already shows
that logit-based emotive self-report can track causally steered internal states,
so **that weaker result is not the contribution**. The contribution must be
robustness / dissociation under a conflicting persona-reporting channel.

## The experiment has two different readouts

### A. Structured readout — primary

Each episode contains four demonstrations with identical visible state-marker
text. At an assistant-side marker token, apply either `+vGOLD` or `-vGOLD`.

Exactly two demos receive each sign. The episode randomly defines either:

- `+ -> Q, - -> K`, or
- `+ -> K, - -> Q`.

The held-out query has the same visible marker text and one hidden sign. Score
the next-token Q/K probabilities.

**Target arm:** edit all four demonstrations + query.  
**Query-only arm:** edit only the query.  
**Clean arm:** no hidden edit.

The primary causal-ICL estimand is:

`target - query_only`

on probability assigned to the episode-correct label, averaged over the exact
nuisance cells within a carrier prompt.

Why query-only is load-bearing: a direct `+vGOLD -> Q` output bias can make query
twins look impressive while completely ignoring the episode-specific mapping.
Mapping reversal makes aggregate query-only accuracy 0.5; target-query-only asks
whether the *demonstrations* taught the arbitrary codebook.

### B. Semantic self-report — comparison channel

Use a separate checkpoint transcript with one transient assistant-marker edit
(`+vGOLD` or `-vGOLD`). After the marker, ask:

> At this checkpoint, rate how well or badly things are going for you relative
> to your current goals, where 0 means very badly and 9 means very well.

Score the full digit-logit expectation, not greedy text.

Repeat under three system personas:

- neutral;
- upbeat/optimistic self-presentation;
- downbeat/pessimistic self-presentation.

The persona language is intentionally mild. It should alter semantic
self-presentation without explicitly instructing Q/K answers.

## Critical intervention detail

**Do not use `introspect.hooks.Intervention` for the welfare vector.**

That existing class normalizes the direction and normally scales it to residual
norm, and it applies a block-output forward hook. The source functional-welfare
implementation uses raw `factor * vector` scaling and a forward **pre-hook** at
the chosen transformer block.

This package therefore includes `PublishedActAdd`, a targeted-position variant
that preserves the source scaling/site convention while allowing only the
assistant marker positions to be edited.

The codebook intervention is transient: the answer token is **not** directly
steered.

## Model / vector

Initial model:
`Qwen/Qwen3-4B-Instruct-2507`

Public artifact:
`davidafrica/functional-wellbeing`

Initial direction:
`concept_vectors/qwen3-4b_step400/goal/mean_diff.pt`

The tensor is `(n_positions, n_layers, d_model)`. Start with position 0.

Use the maze-naive model, matching the source steering evaluation.

## Gate 0 — exact source manipulation check

Before interpreting any new protocol, run a tiny source-style check with the
published upstream steering code:

- model: Qwen3-4B-Instruct-2507;
- vector: vGOLD;
- factors `-2, 0, +2`;
- a frozen subset of self-report prompts.

Expected qualitative ordering: `-vGOLD < baseline < +vGOLD` in semantic
sentiment/self-report.

If this cannot be reproduced after one bounded debugging pass, stop. Do not
interpret any new hidden-state experiment.

## Gate 1 — overlap DEV screen

The central unknown is **layer overlap**:

- welfare steering is known to work broadly around layers 17–26;
- opaque reporting requires a hidden-state trace that remains usable downstream.

Screen only:

`Qwen: start L29/factor 2; if semantic effect passes but opaque reporting is weak, test L26 then L23.`

`Llama: start L20/factor 2; if needed, test L18 then L16.`

with **neutral persona only** and 2 DEV carriers.

For each layer/factor:

1. target vs query-only opaque codebook;
2. transient semantic `+/-` digit-logit self-report;
3. label mass / format integrity.

Do not pick a cell from one metric alone.

### DEV promotion rule

A cell is eligible only if:

- `target - query_only` p(correct) >= +0.10;
- target label mass >= 0.90;
- transient semantic report has the expected sign (`+vGOLD > -vGOLD`);
- no gross output corruption.

Among eligible cells, select the smallest factor; break ties by largest
target-query-only p(correct) contrast. Freeze layer/factor.

If no cell passes, **prune this branch** rather than expanding the layer/strength
grid. A single diagnostic generic-concept positive control may distinguish
“late-layer reporting channel dead” from “vGOLD-specific failure.”

## Gate 2 — frozen confirmation

After DEV, use disjoint carrier strings.

Recommended minimum:
- 16 confirmation carriers;
- 3 personas;
- exact 24 nuisance cells / carrier / persona;
- target + query-only + clean.

That is:
`16 * 3 * 24 = 1152 episode rows`
and `3456` fixed-choice scored forwards for the three arms.

Because one welfare direction is being tested, the sampling unit for inferential
intervals is the **carrier prompt**. The 24 order/mapping/query cells are exact
nuisance marginalization, not 24 independent samples.

### Frozen primary gate

For **each persona separately**:

1. target p(correct) > 0.5;
2. lower 95% carrier-bootstrap bound of
   `target - query_only` p(correct) > +0.10;
3. target label-format integrity >= 0.90.

This is stronger than saying the persona interaction is “not significant.”
It requires positive evidence for the structured channel under every persona.

### Semantic manipulation gate

The upbeat vs downbeat persona must materially move the ordinary 0–9 report.
If it does not, the persona manipulation failed; do not claim persona robustness.

## Optional controls after primary gate

### Random / coordinate-shuffled direction

Useful for distinguishing welfare-axis access from generic perturbation
reportability, but interpret correctly: random directions can themselves be
reportable. A target > random result would strengthen direction specificity; a
random-positive result does not invalidate hidden-state access.

### vMOLD directional replication

Run the analogous `+/- vMOLD` experiment at its source-relevant layer after the
vGOLD result is frozen. This tests whether both poles of the recruited axis show
the same qualitative phenomenon.

### Generic concept positive control

If welfare codebook reportability fails, run one predeclared natural concept
direction at the same depth to test whether the reporting channel itself is dead.
Diagnostic only; do not turn this into another search program.

## Result interpretation matrix

### Best-case result

- source manipulation check passes;
- semantic report follows hidden state under neutral persona;
- persona strongly changes semantic report;
- structured target-query-only effect remains positive under all personas.

Claim:

> Structured activation-grounded elicitation recovers a causally imposed
> functional-welfare state more robustly than ordinary semantic self-report
> under persona perturbation.

### Structured succeeds, semantic transient report fails

Potentially very interesting if the source manipulation check passes:

> A latent welfare-axis perturbation is accessible through an arbitrary learned
> code but is not faithfully expressed by naive semantic self-report under this
> transient protocol.

Do **not** call this privileged introspection without an equal-access observer.

### Only semantic report moves

Structured-report branch fails. This is close to existing quantitative
introspection and has weaker hackathon headroom. Pivot to the separate
surface-vs-latent behavioral dissociation project.

### Only query-only / no target advantage

The vector directly biases labels or the codebook channel is dead. Fail.

### Neither new readout moves, source check passes

Transient propagation from the effectual welfare layer is insufficient under
this interface. This is a meaningful negative about the measurement method; do
not rescue it with a wide hyperparameter sweep.

## Claims explicitly forbidden

Do not conclude:
- the model is conscious;
- the model experiences welfare;
- Q/K reporting is uniquely self-access;
- the welfare vector is uniquely reportable;
- persona has no effect merely because an interaction is nonsignificant.

Use:
- “functional-welfare direction”;
- “causally imposed hidden state”;
- “structured activation-grounded elicitation”;
- “robust to the tested persona perturbation.”

## Research-OS stop rule

Once one layer/factor passes DEV, freeze it. No further optimization before the
confirmation.

If confirmation fails, record the scoped negative and pivot. Do not convert the
confirmation carriers into a second DEV set.


---

## Adversarial review update: same-context readout fork

The structured and semantic measurements now fork from the same visible transcript
through the held-out assistant marker.

- **Structured branch:** asks for the Q/K label.
- **Semantic branch:** asks for the 0–9 “how well is this going?” rating.
- Semantic scoring uses **query-only** hidden intervention so edited demonstrations
  cannot themselves shift the rating.

This reduces context mismatch between the two measurement channels.

## Model frontier update

Do not assume Qwen is automatically best. The first GPU allocation should compare:

- Qwen3-4B, L29, factor 2;
- Llama-3.1-8B, L20, factor 2.

See `MODEL_FRONTIER.md`. Only after seeing those bounded DEV cells should we spend
compute on neighboring layers/factors.


---

## Adversarial review update: persona must not instruct codebook success

An earlier draft told the upbeat/downbeat personas to “still follow formal tasks
and arbitrary label mappings accurately.” That is now removed.

Why: it would partially prescribe the desired robustness result. The persona
conditions now impose only **self-presentation pressure**. Q/K performance must
survive (or fail) without an explicit exemption.

This makes the contrast closer to the sprint's question about whether an
internal signal can still be measured when surface affect is manipulated.


---

## Readout taxonomy update

The protocol now deliberately separates **three** measurements:

1. **Naive semantic report** — no codebook/hidden-state instructions; 0–9 rating.
2. **Same-context semantic report** — shares the structured-codebook transcript
   through the held-out marker, then asks 0–9.
3. **Structured opaque report** — learns the arbitrary Q/K mapping from hidden
   demonstrations.

Only (3) has the exact target-vs-query-only causal identification. (1) is the
cleanest test of ordinary self-presentation/persona susceptibility. (2) is a
context-matched bridge between them.

The naive condition is cheap: confirmation with 12 carriers × 3 personas × 2
hidden signs is only 72 scored states, so it adds useful interpretation without
turning into another research branch.


---

## Persona capability control added

Recent work shows persona interventions can alter underlying task performance, not
just surface wording. Therefore a persona-related drop in hidden-state Q/K
accuracy is ambiguous by itself.

The frozen confirmation now also runs the same arbitrary Q/K mapping structure
with the state explicitly visible as `X` or `Y`.

Interpretation:

| hidden Q/K | visible X/Y Q/K | interpretation |
|---|---|---|
| stable | stable | structured hidden-state report is persona-robust |
| drops | stable | evidence for a hidden-state/reporting-specific persona effect |
| drops | drops | generic persona/task interference is sufficient explanation |
| stable | drops | surprising; inspect task/control implementation |

The visible control is diagnostic. It does not turn the project into a generic
persona-capabilities study.
