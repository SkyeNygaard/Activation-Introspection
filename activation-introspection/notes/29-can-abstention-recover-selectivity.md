# 29 — Can abstention recover the selectivity training destroyed? (pre-run note)

Written before the analysis ran. Nothing above the line was edited afterwards.
**No GPU, no new model run.** This is a secondary analysis of artifacts already on
disk, and that carries its own hazards, handled below.

## Where this came from, disclosed because it matters

[`20`](20-comparator-tiers.md) reported:

> **Calibration is unusable.** Confidence 0.998 when right, 0.928 when wrong. A
> 0.07 gap across a 100% accuracy gap, so confidence cannot filter self-report.

While looking for a live branch I re-opened those 24 rows and computed a different
statistic. The means reproduce exactly. But a difference of means is the wrong
instrument for a number squashed against 1.0. Ranked by confidence instead:

| coverage | accuracy on what is kept |
|---|---:|
| 100% | 0.667 |
| 70% | 0.938 |
| 50% | **1.000** |

AUROC 0.969; the top 12 of 24 are 12 for 12, exact p = 0.00067.

**That is a post-hoc finding on 24 rows and it is not a result.** It is the reason
for this note. `20`'s conclusion may be an artifact of choosing a
difference-of-means test for a variable that cannot spread, and if so the
correction matters, because "the model cannot tell when it is wrong about itself"
is a much stronger claim than the data supports.

## The question this asks instead

The interesting question is not whether `20`'s sentence was sloppy. It is this:

[`08`](08-sensitivity-specificity-tradeoff.md) found that training a model to
report its own states **destroys selectivity** — magnitude-matched random
directions go from 0.513 in the base model to 0.913–0.955 after training. The
trained reporter answers "did something move" instead of "is concept X active".

Anthropic's [Introspection Adapters](https://arxiv.org/pdf/2604.16812) hit the
same wall from a different direction and name the fix as future work they did not
build: reducing the false-positive rate "through improved DPO training,
calibration, or **abstention mechanisms**".

So:

> **If a trained reporter were allowed to say "I don't know", would the
> false alarms on meaningless directions be the ones it declined?**

That is a different question from accuracy, and no introspection paper I have read
asks it. Every one of them reports accuracy and false-positive rate at full
coverage. The mature machinery for this — selective prediction, risk-coverage
curves — exists for question answering and hallucination, and has never been
pointed at a model's reports about its own internals.

It is also the question that decides whether any of this is deployable. A monitor
that is wrong 40% of the time is useless. A monitor that is wrong 40% of the time
**and knows which 40%** is a working monitor with a coverage setting.

## What I am about to do

Analyse `results/report_training_v3_seed{0,1,2,3}_raw.jsonl` — 504 rows per seed,
2,016 total, already on disk, generated under the frozen v3 protocol.

The structure is what makes this worth doing: three arms (`base`, `trained`,
`trained_seen_bank`) crossed with four conditions (`target`, `random`, `shuffled`,
`clean`). Every row carries `correct`, `correct_probability`, `signed_margin` and
`label_mass`. So the same rows that produced `08`'s selectivity finding also carry
a per-row confidence that `08` never used.

Confidence is `abs(signed_margin)` — how far the model's chosen label sat from the
other one. Not a verbalized confidence; an internal margin. That distinction is
stated in the result, because a monitor that needs its own logits is a different
product from one you can ask.

**Three measurements, declared now:**

1. **Ranking.** AUROC of confidence against correctness, per arm, on the `target`
   condition. Does the margin order right above wrong at all?
2. **Risk-coverage.** Accuracy on retained rows as coverage falls from 100% to
   10%, per arm.
3. **The one that matters — selectivity at coverage.** On the `trained` arm, take
   `target` and `random` together and ask: as coverage falls, does the gap between
   them reopen? `08`'s finding is that at full coverage there is no gap. If the
   random-direction responses are the low-confidence ones, abstention restores the
   discrimination training removed, and that is a usable fix for a named open
   problem. If they are just as confident, abstention cannot help and the
   trade-off is structural.

## Development and confirmation, split before looking

**Seeds 0 and 1 are development. Seeds 2 and 3 are confirmation.** Declared here,
before any of the four is opened.

This is the part that needs care, and it is why the note exists. These artifacts
are **already published** — `v3` is the citable four-seed training result. Reusing
them for a new question is legitimate, because this is a different estimand from
the one they were frozen for. But every threshold, every coverage level, and every
choice about which arm to feature must come from seeds 0 and 1 only, and then be
reported unchanged on seeds 2 and 3.

Reporting a number chosen after seeing all four seeds would be exactly the error
[`13`](13-shared-axis-audit.md) and [`15`](15-matched-reader-on-content.md) made,
with the extra aggravation that the data were lying around and cost nothing.

**The `20` pilot is development too**, and it is now spent. It cannot be quoted as
evidence for anything this analysis concludes.

## What each outcome would mean

**Ranking works and the random-direction false alarms are low-confidence.** The
strongest outcome. Training destroys selectivity in the reported label, but the
information survives in the margin, and abstention recovers it. That is a concrete
answer to Introspection Adapters' stated open problem, it corrects `20`, and it
changes `08` from "training breaks the monitor" to "training breaks the monitor's
*output*, and the fix is to let it abstain".

**Ranking works but random directions are just as confident as targets.** Then
`20`'s headline is still wrong — the model does know when it is wrong about *which*
concept — but abstention does not fix specificity, and `08`'s trade-off is
structural. Still a real finding, and a more pessimistic and more useful one for
anyone proposing self-report as an audit.

**Ranking fails on the trained arm but works on base.** Training destroys the
confidence signal as well as the label. That would be the sharpest version of
`08`: training doesn't just make the reporter wrong, it makes it wrong
*confidently*, which is the worst property a monitor can have.

**Ranking fails everywhere.** `20`'s conclusion stands, the 24-row pilot was noise,
and this branch closes. Cheap, and it removes a lead I would otherwise keep
returning to.

## Kill rule

If AUROC on the development seeds is below 0.60 for both `base` and `trained` on
the `target` condition, stop. `20` was right, the pilot was 24 rows of luck, and no
coverage analysis is worth reporting on a signal that does not rank.

## Prediction, on the record

I expect **ranking to work** — AUROC 0.75–0.90 on both arms — because the pilot
was strong and margins usually carry this much.

I expect **abstention not to fix selectivity**. Roughly 65/35. `08`'s reading is
that the trained model has become a displacement detector, and a displacement
detector should be *confident* about a random direction: something really did move.
If that is right, the random-direction rows will sit at high margin, and the
`target`/`random` gap will not reopen as coverage falls.

If that is what happens, the honest headline is uncomfortable and worth having:
**letting the model abstain makes it more accurate about which concept, and no
better at telling a concept from noise.**

## Cost

Minutes of CPU. No model load, no GPU — which is also why it is being done now:
another session on this machine is holding the GPU, and this branch needs none of
it.

The real cost is the artifacts. These four seeds are a published, frozen result and
this analysis spends two of them as development. That is the price, it is paid
deliberately, and it is recorded here so nobody later mistakes seeds 0 and 1 for
held-out data on this question.

## What would change my mind about running it at all

If `signed_margin` turned out to be a coarse or degenerate quantity — heavily tied,
or a rounded value with few distinct levels — then no ranking analysis is
meaningful and the right move is to re-run and save a proper probability rather
than to analyse a stub. Checked first, reported in the result.

---

## Amendment, after the development pass and before any confirmation

**I picked the wrong artifact for the third measurement, and I am saying so here
rather than quietly switching.**

The development pass on `report_training_v3` seeds 0 and 1 ran as planned, the
margin is healthy (119 distinct values, no mode above 8.2%), and the kill rule did
not fire. Measurements 1 and 2 are answerable there and will be confirmed on seeds
2 and 3 exactly as declared.

But measurement 3 is not answerable there. **`report_training_v3` does not contain
[`08`](08-sensitivity-specificity-tradeoff.md)'s selectivity loss.** Its `trained`
arm sits at 0.52 on random directions — chance, the same as base. `08`'s
0.913–0.955 comes from `remap_training_v1/v2`, whose arms are named `fixed` and
`remap`, and I matched on the word "trained" instead of checking the source. The
note above says "the same rows that produced `08`'s selectivity finding"; that
sentence is wrong and it is left in place.

So the third measurement moves to `results/remap_training_v2_seed{0,1,2}_raw.jsonl`
— 4,608 rows per seed, arms `base`/`fixed`/`remap`, conditions `target`/`random`,
three strengths, `signed_margin` present. `08`'s table is the strength-0.5 cell, so
that is the cell analysed.

**Declared now, before opening any of it: seed 0 is development, seeds 1 and 2 are
confirmation.** Everything else — the coverage grid, the AUROC definition, what
each outcome means, the prediction — carries over unchanged from above.

Two things this costs, recorded so they are not forgotten. The prediction I made
was written with the wrong artifact in mind, though it happens not to depend on
which one — it is about displacement detectors being confident, and `remap` is the
displacement detector. And `report_training_v3` seeds 0 and 1 are now spent on
measurements 1 and 2, while `remap_training_v2` seed 0 is spent on measurement 3.
Neither can be reused as held-out data for these questions.

---

# Result: it knows when it is wrong, and that does not help at all

Run **2026-08-12**. No GPU, no model load — analysis of frozen artifacts.
Summary: `results/selective_introspection_v1_summary.json`. Script:
`scripts/analyze_selective_introspection.py`.

The margin is a usable quantity: 119 distinct values in `report_training_v3`, 181
in `remap_training_v2`, no mode above 8.2% and 1.5% respectively. Not a stub.

## 1. `20`'s conclusion is wrong, and it confirms

`20` said confidence cannot filter self-report, on the strength of a 0.07 gap
between mean confidence when right and when wrong. Ranked instead of averaged, on
data `20` never touched:

| arm | accuracy, answer everything | AUROC | at 70% coverage | at 50% | at 30% |
|---|---:|---:|---:|---:|---:|
| `base` | 0.479 | **0.675** | 0.522 | 0.583 | 0.690 |
| `trained` | 0.969 | **0.891** | **1.000** | **1.000** | **1.000** |

Development AUROCs were 0.675 and 0.958; confirmation gives 0.675 and 0.891 on
seeds never opened. **The signal is real and it replicates.**

So: *the model does know which of its self-reports to distrust.* `20` measured a
difference of means on a quantity squashed against its ceiling and concluded the
signal was absent. It was there the whole time, in the ordering. **`20`'s
"calibration is unusable" is withdrawn.**

The caveat that keeps this honest: confidence here is the model's internal logit
margin, not a number it says. A monitor built on it needs activation access, which
is a different product from one you can simply ask — and it puts this squarely
back inside the cost criterion the rest of this repository lives under.

## 2. And abstention does not fix selectivity. It makes it worse.

This is the measurement the note exists for, and the answer is sharper than the
question deserved.

`08` found training destroys the model's ability to tell a real concept from a
magnitude-matched random direction. If the false alarms were the *unconfident*
ones, letting the model abstain would recover it. Gap between accuracy on real
concepts and accuracy on random directions, as the least-confident rows are
dropped — confirmation seeds, 768 rows per cell:

| arm | answer everything | 70% | 50% | 20% |
|---|---:|---:|---:|---:|
| `base` (untrained) | 0.232 | 0.260 | 0.289 | **0.455** |
| `fixed` (trained) | 0.059 | 0.034 | 0.023 | **0.013** |
| `remap` (trained) | 0.099 | 0.037 | 0.029 | **0.019** |

**The two directions are opposite, and both replicate from the development seed.**

For the **untrained** model, abstention works exactly as one would hope: keep only
the confident half and the separation between meaningful and meaningless nearly
doubles, 0.232 to 0.455. Its mistakes about noise are the ones it is unsure about.

For the **trained** reporters, abstention runs backwards. The gap collapses toward
zero — 0.059 to 0.013, and 0.099 to 0.019. **The rows where a trained reporter
confidently reports a concept are disproportionately the rows where the injected
direction was meaningless.** It is not merely that training makes the monitor
wrong. Training makes it wrong exactly where it is most certain, which is the worst
possible arrangement of those two properties.

The base arm reproduces `08` to three decimals — target 0.745, random 0.513 — so
this is the same effect `08` measured, seen through a lens `08` did not have.

## 3. What did not confirm

Development suggested the confidence margin might itself separate real concepts
from random directions even when the label cannot — AUROC 0.815 for `fixed` and
0.769 for `remap` against 0.570 for base. That looked like selectivity surviving
in the margin after being destroyed in the label.

**It does not hold up.** Confirmation gives 0.612 for `fixed` and 0.722 for
`remap`. `fixed` is barely above the base model's 0.570. One of the two arms lost
most of the effect, so this is reported as **failed to confirm**, not as a weaker
version of itself. It was the most interesting-sounding thing in the development
pass, which is why the split existed.

## My prediction, scored

I predicted ranking would work at AUROC 0.75–0.90: **half right.** The trained arm
landed at 0.891, inside the range; the base arm at 0.675, below it.

I predicted 65/35 that abstention would not fix selectivity, on the reasoning that
a displacement detector should be confident about a random direction because
something really did move. **That was right, and the mechanism was right, and the
effect is larger than "does not fix"** — abstention actively removes the residual
discrimination. I did not predict that the untrained model would go the other way,
and that contrast is the most useful thing here.

## What this establishes

Three things, in descending order of confidence:

1. **A model's confidence in its own self-report carries real information about
   whether that self-report is correct**, at AUROC 0.675–0.891, replicated. This
   withdraws `20`.
2. **Abstention cannot repair a trained introspective monitor.** Anthropic's
   [Introspection Adapters](https://arxiv.org/pdf/2604.16812) names "abstention
   mechanisms" as one of three candidate fixes for the false-positive rate that
   limits their method. On this setup, at this scale, that candidate fails — and
   fails in the specific way that matters, because the false positives are the
   confident ones.
3. **Training inverts the sign of the abstention benefit.** Untrained, filtering by
   confidence nearly doubles the concept-versus-noise separation. Trained, it
   erases it. That is a property of introspection training nobody has reported,
   and it is measurable in an afternoon on saved rows.

## Limits

One model, one layer, one training recipe, one strength cell (0.5), one bank. Three
seeds for the selectivity measurement and four for the ranking, all from runs
frozen for a different purpose — this is a **secondary analysis**, and the arms
were not designed to answer this question. Confidence is an internal margin, not a
verbalized report. The adapters were never saved, so nothing here can be re-scored
against a fresh training run.

Most importantly: this shows abstention fails *for this recipe*. It does not show
abstention fails for the DPO-refined adapters Anthropic actually proposes, which
are trained specifically to prefer accurate reports over plausible ones. That is
the obvious next experiment and it is not one this machine can run.
