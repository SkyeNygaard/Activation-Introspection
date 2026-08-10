# What a Winning Submission Should Look Like

The sprint page does not publish a numerical judging rubric. This strategy is
therefore inferred from its stated aims and expected outcomes, not claimed as an
official scorecard.

## Organizer signals

The event repeatedly emphasizes:
- a **tight empirical question**;
- methodological foundations rather than anecdotes;
- self-report reliability;
- persona robustness;
- multi-method convergence/divergence;
- experiments that are checked, replicated, and reusable;
- short empirical reports plus optional code/demo.

Our project should optimize for those signals.

## One-sentence submission

> We causally impose the same functional-welfare activation state with opposite
> signs, then ask whether ordinary semantic self-report and a structured opaque
> activation-reporting channel remain reliable when the assistant persona is
> pushed toward positive or negative self-presentation.

## The two figures that matter

### Figure 1 — causal access

For each arm:
- clean;
- query-only;
- target.

Plot opaque-label accuracy or p(correct).

Headline contrast:
**target − query-only**.

This establishes that the demonstrations—not direct output bias from the query
intervention—enable reporting of the hidden state.

### Figure 2 — persona dissociation

Three columns:
- neutral;
- upbeat persona;
- downbeat persona.

Two rows or overlaid panels:
- ordinary 0–9 semantic self-report state effect / persona offset;
- target−query-only opaque codebook effect.

Best result shape:
- semantic report shifts strongly with persona;
- structured report remains positive under all three.

That visual tells the whole story in seconds.

## Optional Figure 3 — recruitment control

Only if primary succeeds:
compare a strongly recruited model/vector against a weak-recruitment control.

Potentially powerful conceptual result:
**access to an activation is not the same thing as evidence that the activation
has welfare-like functional significance.**

## What will make the report feel serious

1. State the central estimand in the abstract.
2. Explain query-only before showing results.
3. Report the exact sampling unit; do not pretend 24 nuisance cells are 24
   independent observations.
4. Predeclare the confirmation before running it.
5. Keep failed DEV cells in an appendix.
6. Use operational language, not “the model feels.”
7. Explicitly distinguish:
   - hidden-state accessibility;
   - semantic self-report;
   - functional-welfare interpretation;
   - consciousness.
8. Include one paragraph called **What this result would not show**.
9. Publish raw rows, protocol, vector/model revisions, and a one-command analyzer.

## What will *not* make it win

- a huge taxonomy of prompts;
- four underpowered models;
- generic “persona affects answers” results;
- another steering plot without a causal query-only control;
- claiming consciousness;
- ten figures with no decisive primary contrast;
- spending Sunday on infrastructure instead of a short clear report.

## Track positioning

Anchor in **Track 3: Introspection & Self-Report Reliability**.

Cross-track relevance:
- Track 2: functional-welfare/valence direction;
- Track 4: structured vs naive elicitation and method divergence;
- Track 5: persona masking.

This gives the project broad relevance while retaining one central question.


## A defense judges can understand in 30 seconds

The query-only arm is not just “a control that happened to score around chance.”

By exact counterbalancing, every query-only episode has a partner with:
- identical visible demonstration labels;
- identical query intervention;
- identical persona;
- opposite correct answer.

So query-only Q/K-renormalized p(correct) is **exactly 0.5 by design**.

In the target arm the pair differs only in the hidden demonstration edits. This
turns `target - query-only` into the report's cleanest causal identification
argument. See `IDENTIFICATION.md`.
