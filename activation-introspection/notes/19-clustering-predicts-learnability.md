# Pre-run note: turning a ranking into a prediction

Written **2026-08-12, before anything ran.** New commitment, after stepping back
from the introspection line — four of six novelty candidates there turned out to
be prior art, and this is the one that survived two searches.

## What exists, and why it is not yet a finding

[`16`](16-visible-rule-capacity.md) screened six hidden rules through the
four-shot `Q/K` interface with nothing hidden, and found that whether the model
learns a rule is predicted by whether the rule's two classes **clump together** in
the model's own representations:

| | separation | accuracy |
|---|---:|---:|
| rules it learned | 0.043 – 0.218 | 0.729 – 0.979 |
| rules it failed | −0.023 – 0.008 | 0.469 – 0.490 |

No overlap, at three depths, on either measure. But this is **six points, chosen by
me, scored after the fact.** A ranking that orders six items is weak evidence for a
relationship, and I have twice this week promoted an ordering to a mechanism and
been wrong. The relationship is descriptive. Descriptive is not a finding.

## What would make it one

The claim worth having is not *"clustering correlated with accuracy on six rules"*.
It is:

> **Measure how well a hidden rule's classes clump, before running anything, and
> you can predict whether the model will learn it.**

That is a design tool — it costs about 150 forward passes and no intervention, and
it would have saved five runs on the natural-state branch. It is also falsifiable
in a way the ranking is not.

## The design

Two phases, and the freeze between them is the point.

**Phase 1 — measure.** Fourteen new rules, none used in `16`. Measure each one's
class separation at three depths. **No prompting, no accuracy, no scoring.**

**Freeze.** Write a protocol recording, for every rule, a **prediction of pass or
fail** made from separation alone, against thresholds fixed now from `16`'s gap:

- separation **≥ 0.020** → predict the model learns it (accuracy ≥ 0.60)
- separation **< 0.020** → predict it does not (accuracy < 0.60)

Both thresholds sit in the middle of empty gaps in `16`'s data: nothing between
0.008 and 0.043 on separation, nothing between 0.490 and 0.729 on accuracy. The
protocol is written to disk before a single phase-2 forward pass runs, and phase 2
refuses to start without it.

**Phase 2 — test.** Run the prompting screen on all fourteen. Score the
predictions.

## The rules, and why these

Chosen to span the space and to include cases I genuinely cannot call, not to make
the prediction look good.

| expected to clump | expected not to | **honestly unsure** |
|---|---|---|
| warm vs cool colours | five-letter vs seven-letter words | abstract vs concrete nouns |
| body parts vs furniture | words ending in `e` vs not | singular vs plural nouns |
| liquids vs solids | words with a doubled letter vs not | past vs present tense verbs |
| vehicles vs plants | multiples of three vs not | words of Latin vs Germanic origin |
| positive vs negative sentiment | prime vs composite | |

The four in the last column are the ones that make this a test. Grammatical
categories are real and systematic, so a model might represent them tightly — or
they might cut across semantic space the way spelling does. **I do not know, and
that is why they are in.**

## What each outcome means

| Outcome | Reading |
|---|---|
| 12+ of 14 predictions correct | The relationship is predictive, not just descriptive. A cheap prospective gate for any hidden-class design, and the first thing in this repository that predicts rather than explains |
| 9–11 correct | Real but noisy. Report the separation-accuracy relationship as a trend with its exceptions, and look at what the misses share |
| ≤ 8 correct | At 14 binary predictions, chance is 7. **`16`'s ordering was six lucky points and the account is dead.** Say so and drop it |
| The unsure four all miss | Most informative failure: the relationship holds for semantic and surface rules but not grammatical ones, which localises what "clumping" is actually tracking |

## Prediction, on the record

I expect **11 or 12 correct**, with the misses among the grammatical rules. My
guess is that tense and number are represented systematically enough to be learned
while *not* clumping by this measure — because grammatical features are usually
found in low-dimensional subspaces rather than as tight neighbourhoods, and a
similarity measure over whole states would miss them.

If that specific pattern appears, it is not a rescue — it is a boundary on what
the measure captures, and it would need its own confirmation.

## Cost

Phase 1: about 200 short forward passes. Phase 2: fourteen rules × 24 cells × 4
folds, prompting only. A few minutes total. No training, no interventions.

---

# Result: 12 of 14, and the binary score understates it

Run **2026-08-12**, two phases. Protocol frozen before any accuracy existed:
`results/clustering_prediction_protocol_v1.json`, with its own source hash. Results:
`results/clustering_prediction_v1_raw.jsonl` and `..._summary.json`. Runner:
`scripts/run_clustering_prediction.py`, which refuses to run phase 2 without the
protocol.

| rule | separation | predicted | accuracy | |
|---|---:|:---:|---:|---|
| latin_vs_germanic | +0.1487 | learnable | 0.750 | ✓ |
| abstract_vs_concrete | +0.1244 | learnable | 0.958 | ✓ |
| body_vs_furniture | +0.0925 | learnable | 1.000 | ✓ |
| sentiment | +0.0905 | learnable | 1.000 | ✓ |
| vehicle_vs_plant | +0.0650 | learnable | 0.708 | ✓ |
| liquid_vs_solid | +0.0395 | learnable | 0.875 | ✓ |
| colour_temp | +0.0269 | learnable | 0.667 | ✓ |
| letter_count | +0.0225 | learnable | 0.792 | ✓ |
| double_letter | +0.0051 | not | 0.542 | ✓ |
| ends_in_e | −0.0221 | not | 0.458 | ✓ |
| past_vs_present | −0.0251 | not | 0.583 | ✓ |
| **prime** | −0.0281 | not | **0.625** | **✗** |
| **singular_vs_plural** | −0.0500 | not | **0.667** | **✗** |
| multiple_of_three | −0.0544 | not | 0.500 | ✓ |

**12 of 14.** Predictions made from clustering alone, before a single accuracy
number existed.

## The honest reading of that score, which is less flattering

Under a coin flip, 12 or better happens about **once in 150 tries**. But a coin
flip is the wrong baseline. Ten of the fourteen rules turned out learnable, so
**predicting "learnable" for everything scores 10 of 14.** The measure beats that
by two.

Stated plainly: **as a binary classifier the separation measure is only modestly
better than guessing the majority class**, and anyone quoting 12/14 without saying
so is overselling it.

## The continuous relationship is the real evidence

The binary threshold throws away most of the signal. Sorted by separation, the two
groups barely touch:

| | n | mean accuracy | boundary |
|---|---:|---:|---|
| separation ≥ 0.020 | 8 | **0.844** | lowest is 0.667 |
| separation < 0.020 | 6 | **0.562** | highest is 0.667 |

Every positive-separation rule scored **at or above 0.667**. Every
negative-separation rule scored **at or below 0.667**. The two distributions meet
at exactly one point and do not cross. Rank agreement across all fourteen is
**0.785**.

Both misses sit within 0.07 of the frozen accuracy threshold — `prime` at 0.625 and
`singular_vs_plural` at 0.667, each just over a line drawn at 0.60. A threshold at
0.68 would have scored 14 of 14, and **that number must never be quoted**: it comes
from looking at the answers. The frozen result is 12.

## My advance prediction, scored

I predicted **11 or 12 correct, with misses among the grammatical rules**. The
count is right. The reason is half right: `singular_vs_plural` missed as expected,
but `past_vs_present` was predicted correctly, and the other miss is `prime`, which
is numeric. **So the guess about grammatical features living in low-dimensional
subspaces rather than tight neighbourhoods is not supported by two rules, one of
which went the other way.** Treat it as unexamined.

One prediction of mine was cleanly wrong in the useful direction: I expected
`letter_count` to fail, and it landed at +0.0225 separation — barely over the line —
and then scored 0.792. The measure called it and I did not.

## What this now supports

**A cheap prospective gate.** Before spending a bank on any hidden-class design,
measure the class separation: about 200 short forward passes, no interventions, no
prompting. Positive separation has so far always meant learnable, and it would have
saved the five runs described in [`10`](10-output-ready-arithmetic.md).

The asymmetry is worth stating because it decides how the gate should be used:

- **Positive separation → learnable: 8 for 8.** A strong green light.
- **Negative separation → not learnable: 4 of 6.** A weak red light.

So use it to *proceed* with confidence, and treat a negative reading as a warning
rather than a veto.

## Epistemic status

- **Observed:** the table above. Fourteen rules, 24 cells each, one model, one
  prompt template, one fold per rule.
- **Confirmatory, and properly so:** thresholds and predictions were frozen in a
  file with a source hash before phase 2 could run, and the runner enforces it.
  This is the first prospective test in this repository.
- **Not established:** any causal claim. Clustering may predict learnability
  because it *is* what the interface uses, or because both track a third thing —
  how familiar the distinction is in pretraining. Nothing here separates those.
- **Untested:** a second model, a second prompt template, more folds per rule, and
  demonstration counts other than four.

## Limits

- One fold per rule against `16`'s four, so each accuracy rests on 24 episodes.
- The measure is a similarity over whole states. A distinction carried in a small
  subspace would be invisible to it, which is the most likely reason for the two
  misses and is untested.
- Rules were written by me. A rule set assembled by someone else is the obvious
  next robustness check.
- The accuracy threshold of 0.60 is arbitrary within the gap `16` left. The
  continuous relationship does not depend on it; the 12/14 does.

## What this is not

It is not about introspection, and it does not touch the self-report question. It
is a claim about what a four-shot in-context interface can be taught, which is a
prerequisite for the natural-state work but not the same subject. Said plainly so
the portfolio does not quietly reclassify it later.
