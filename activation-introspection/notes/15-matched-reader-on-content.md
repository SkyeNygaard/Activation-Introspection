# Pre-run note: does the content result survive the cost criterion?

Written **2026-08-12, before the run.**

## Why this is the run that has to happen next

[`14`](14-content-versus-disturbance.md) established that the model discriminates
two different injected concepts at 0.899 against 0.594 for random directions at
matched separation. That is a **capability** claim. It is not an introspection
claim, and the note says so in as many words.

[`11`](11-matched-cost-reader.md) supplies the field's operative test: a process
counts as introspection only if it is more reliable than one *"with equal or lower
computational cost available to a third party"*
([arXiv 2508.14802](https://arxiv.org/abs/2508.14802)). On the polarity task the
model failed that test completely — a four-shot nearest-centroid reader scored
1.000 against the model's 0.892, with 62 reader-only episodes and **zero**
model-only episodes.

**That result now has a hole, and [`13`](13-shared-axis-audit.md) is what opened
it.** The polarity task collapses to the sign of a projection onto one shared
axis. So `11` measured the cost criterion on a task that never required content —
exactly the objection `13` raises against everything else in this repository. Until
the same comparison runs on a task that does require content, `11` is a result
about a degenerate task and `14` is a result that cannot be named.

One run closes both holes.

## What I am about to do

Re-run `14`'s episodes and capture the five post-injection states at the marker
positions **in the same forward pass that scores the model**, then fit a four-shot
nearest-centroid reader on the four demonstration states and ask it for the fifth.
The reader is imported from `run_matched_reader.py`, not reimplemented.

Both tasks run in the same process on the same carriers and pairs, so the
polarity/content comparison is internal and not a cross-run inference — which is
the defect `05` had to retract a headline over.

| arm | classes | what it gives |
|---|---|---|
| `content` | `v_A` vs `v_B` | the question |
| `polarity` | `+v` vs `−v` | reproduces `11`'s task inside this run, on these pairs |
| shuffled-label reader | labels permuted | the reader is using labels, not geometry alone |

Comparison is **paired**: the same episode is scored both ways, so the four-cell
table of model-correct × reader-correct is the result, not the difference of two
means. That table is what made `11` decisive, and a margin alone would not be.

## What I expect, said before the run so it cannot be reframed after

**I expect the reader to win again**, and probably by a lot. The two concept
directions are close to orthogonal, the reader gets four demonstrations of the
same pair it is queried on, and a two-centroid comparison is close to optimal for
that. Predicting the flattering outcome and then reporting the unflattering one is
how `11` and `12` went; predicting the unflattering one here is not modesty, it is
what the geometry says.

## What each outcome means

| Outcome | Reading |
|---|---|
| Reader dominates on content as it did on polarity | The central thesis of this repository generalizes from a degenerate task to one that provably requires content. `11` stops being vulnerable to `13`, and `14` is named correctly: a real capability that still loses to a cheap outsider |
| Reader wins on polarity but **not** on content | The most interesting outcome available. It would mean the model has an advantage precisely where content is required, which is the first privileged-access signal in this repository and would need replicating before anything else |
| Reader fails on both | Something is wrong with state capture. Stop and fix; read nothing into the model |
| Model-only episodes appear at all | Worth reporting whatever the totals say. `11` found exactly zero in 576, and any non-zero count on a content task is a different phenomenon |

## What it costs

576 episodes across two arms, states captured in the passes that already run.
Roughly seven minutes at today's measured rate. No training.

## What this still cannot do

It says nothing about naturally computed states. It is one model, one layer, one
strength, four concept pairs. And a reader that reads content off the residual
stream is not evidence that the *content* is what the model uses — only that the
information is there and cheaply available, which is the same limit every result
in this repository carries.

---

# Design expanded before the run, after external review

An external review of this repository, received **while this script was being
written and before it was run**, made two criticisms. Both are correct, both are
recorded here rather than quietly absorbed, and the second changes this
experiment's design.

## Error 1: the 28/28 positive-cosine claim is vacuous

`run_bank_audit.py` computes within-bank overlap as

```python
_spread([abs(v) for v in pairwise_cosines(train_bank).values()])
```

and `_spread` then reports `n_positive` as the count of values above zero. **The
absolute value is taken first, so every non-zero entry is positive by
construction.** The statement "all 56 pairs of concept directions are positive,
where about half should be negative" appears in [`13`](13-shared-axis-audit.md),
`CLAIMS.md` and `RESEARCH-DIRECTION.md`, and **the artifact does not support it.**

The shared-axis conclusion does not rest on it. What does support it, unchanged
and measured without any absolute value:

- the reader weight overlaps the average training direction at 0.99999;
- the reader weight has a **positive** overlap with **all eight** held-out
  directions, 0.171–0.243, and those numbers were never absolute-valued;
- the average training direction and average held-out direction overlap at 0.480;
- the average of the unit training directions has length 0.451 where evenly spread
  directions give 0.354 — and that arithmetic implies a mean *signed* overlap of
  about 0.090, close to the measured mean magnitude of 0.096, which is only
  possible if most pairs really are positive.

That last point is an inference from a valid measurement, not a measurement. **The
signed cosines are therefore recomputed in this run** and the claim will be
restated at whatever the signed numbers actually say.

I also over-stated one term. Calling `1/sqrt(2048) = 0.022` the "chance cosine" is
loose: the expected signed overlap between two random directions is **zero**, and
0.022 is the typical size of the departure from it. Conclusions unaffected;
wording corrected.

## Error 2: the off-axis comparison used a handicapped comparator

This is the substantive one. [`13`](13-shared-axis-audit.md) concluded that
training "buys per-episode calibration, which a once-fitted probe cannot do by
construction", and treated the fixed probe's collapse on random directions as
evidence that training provides generality probing cannot.

**The literal statement is true and the conclusion drawn from it does not follow.**
The trained reporter in [`08`](08-sensitivity-specificity-tradeoff.md) receives
four episode-specific demonstrations and a re-randomised Q/K convention on every
evaluation episode. The fixed probe in `13` receives none of that — it is one
weight vector fitted once on a different bank. They do not have equal access, so
the comparison cannot support a claim about what probing can do in general.

The right comparator already exists in this repository: **`11`'s four-shot reader,
which refits centroids inside every episode from that episode's own demonstration
states.** That reader can recalibrate exactly as the trained model can, and there
is no obvious reason it would fail on a random direction — the four demonstrations
are injected with the same random direction as the query.

So `13`'s reversal repeated the error it had just diagnosed in `12`: it fixed one
badly matched comparison and immediately made another one level up. That is the
pattern worth naming, and it is why this run is expanded rather than shipped as
written.

## The arms, revised

| arm | classes | strength | question |
|---|---|---:|---|
| `content` | `v_A` vs `v_B` | matched | can `14` be named? |
| `polarity` | `+v` vs `−v` | 1.000 | reproduces `11` internally |
| **`random_polarity`** | `+r` vs `−r` | 1.000 | **does the adaptive reader keep the generality I credited to training?** |
| **`polarity_weak`** | `+v` vs `−v` | **0.150** | the regime where the base model is blind at 0.500 and training reaches 0.79–0.86 |

**Predictions, before the run.** I expect the adaptive reader to score near 1.000
on `random_polarity`. If it does, `13`'s claim that training buys generality a
probe cannot have is **refuted**, and the honest statement shrinks to: training
buys generality a *fixed* probe cannot have, which is a much weaker and less
interesting thing to say. I expect the reader to beat the model on
`polarity_weak` too, though with less confidence — a 0.15-strength edit is a small
displacement and the reader may degrade.

## What is blocked, and why

The review's preferred experiment is the four-shot reader run **paired against the
trained adapters** on `08`'s own episodes. That cannot run: **the adapters were
never saved.** Only their scores were kept. This is why `12` and `13` both compare
against published numbers "rather than adapters re-run in this process" — a
limitation both notes state without explaining, and the explanation is that
re-running them is impossible without retraining.

That is a genuine reproducibility gap and it is recorded as one. Retraining is
ruled out by a standing decision to spend nothing further on LoRA, so the
adapter-paired comparison stays open. **What this run can settle without any
adapter is whether the cheap adaptive reader has the generality in question** — and
if it does, the adapter comparison is no longer needed to retract the claim.

---

# Result: the reader dominates everywhere, and note 13's reversal is dead

Run **2026-08-12**. 1152 episodes, 342 seconds. Artifacts:
`results/matched_reader_content_v1_raw.jsonl`,
`results/matched_reader_content_v1_summary.json`. Runner:
`scripts/run_matched_reader_content.py`.

| arm | model | four-shot reader | model − reader | both | **model-only** | reader-only |
|---|---:|---:|---:|---:|---:|---:|
| `content` | 0.899 | **1.000** | −0.101 | 259 | **0** | 29 |
| `polarity` | 0.917 | **1.000** | −0.083 | 264 | **0** | 24 |
| `random_polarity` | 0.663 | **1.000** | −0.337 | 191 | **0** | 97 |
| `polarity_weak` (0.15) | 0.497 | 0.833 | −0.337 | 129 | **14** | 111 |

## The result that matters most

**The adaptive reader scores 1.000 on random directions.** The fixed probe in
[`13`](13-shared-axis-audit.md) scored 0.479 on the same kind of directions.

So `13`'s conclusion — that training buys generality a probe cannot have — is
**refuted**. The generality belongs to *per-episode adaptation*, not to training.
A four-shot nearest-centroid comparison, which is the cheapest reader that can use
labels at all, has it in full. What `13` actually demonstrated is that a **fixed**
probe lacks it, which is true, uninteresting, and not what the note claimed.

The external review that predicted this was right, and the mistake is exactly the
one `13` had just finished diagnosing in `12`: fix a badly matched comparison,
then immediately make another one level up. Naming the pattern did not prevent
repeating it.

## The weak-strength arm, which is new information

At strength 0.15 the model is at **0.4965** — reproducing `08`'s base-model
reading of "exactly 0.500, blind" almost exactly, which is a useful internal
check on the harness. The reader is at **0.833**.

`08` puts the *trained* adapters at **0.790–0.863** in this regime. The reader is
at 0.833, inside that range. So training lifts the model from blind to roughly
where a two-centroid comparison already sat — **it closes the gap to the reader
without exceeding it.** That is the third of the four outcomes named before the
run, and it is the one most directly relevant to Project 1.

This comparison is indirect: `08`'s adapter numbers come from a different bank and
carrier set and the adapters cannot be re-scored, so this is two numbers from two
runs, not a paired test. It is suggestive, not decisive, and is labelled that way.

**The 14 model-only episodes are the first in this repository.** Across `11`'s 576
episodes and the three strong arms here, the count of episodes where the model
succeeds and the cheap reader fails is exactly zero. At weak strength it is 14 of
288 — still swamped by 111 reader-only episodes, but no longer a clean dominance
relation. If any regime holds a privileged-access signal, it is this one, and it
would need its own design to chase.

## Corrected: the signed cosines

Recomputed without the absolute value that made the original count vacuous:

| statistic | value |
|---|---:|
| pairs measured | 28 |
| **positive, signed** | **28** |
| mean signed | 0.0975 |
| mean magnitude | 0.0975 |
| minimum signed | **+0.0119** |
| maximum signed | +0.1825 |

The mean signed value equals the mean magnitude and the minimum is positive, so
**every pair really is positive.** The method in `run_bank_audit.py` was vacuous;
the claim it was used to support is true, and is now measured properly rather than
accidentally. Both facts are recorded — a conclusion that survives a broken
measurement was still not established by it.

## The shuffled-label control, and a disclosed defect

Shuffled-label readers scored 0.576, 0.580, 0.601 and 0.417 across the four arms,
against a correct expectation of 0.500. The fair readers sit at 1.000, so the
control does its job — it shows the reader is using the labels and not geometry
alone.

But the spread is wider than it should be, and the reason is a defect of mine:
the permutation is seeded by episode index, giving **24 distinct permutations
reused across every pair and carrier**, so the effective independent unit is the
permutation rather than the episode. At roughly 24 independent draws the standard
deviation at 0.5 is about 0.10, which puts every observed value inside one
standard deviation. This is the same defect `11` disclosed in its own v2 smoke and
I reproduced it. No gate turns on the control's exact value, but it should not be
quoted as evidence of a well-calibrated null.

## What the branch now says, all together

| task | what the model must do | model | cheap reader |
|---|---|---:|---:|
| polarity | pick a side of one axis | 0.917 | 1.000 |
| content | tell two concepts apart | 0.899 | 1.000 |
| random | pick a side of a meaningless axis | 0.663 | 1.000 |
| weak | detect a small displacement | 0.497 | 0.833 |

**One sentence: information sitting in the residual stream exceeds what the model
uses — on planted axes, on semantic content, on meaningless directions, and at
strengths where the model is behaviourally blind — and in 1440 episodes across
four task structures there are exactly 14 where the model succeeds and a
two-centroid comparison fails.**

That is now the branch's result, and it no longer depends on any single design.

## Limits

- The trained adapters could not be included, because they were not saved. Every
  statement here about training is a comparison of two numbers from two runs.
- One model, one layer, four concept pairs, one random seed per control direction.
- The reader is scored only on the state-to-label decision; the model also parses
  the prompt and emits a formatted token, which it does at format rate 1.000.
- The reader reads at the injection site. `11` tested that objection across all
  depths for the polarity task and found the reader perfect over 25 consecutive
  blocks; that sweep has not been repeated for content.
