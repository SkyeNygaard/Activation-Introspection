# 38 — Does introspection training report concept identity, or the fact of displacement? (pre-run note)

Queued **2026-08-14**. Not run.

**This is not a notes/29–37 descendant.** The handoff bans another prompt-conflict
variant and it is right to. This changes the intervention, not the wording: it is
the first note here that acts on *training* rather than measuring its output.

## The question

Every study in this repository measures what a trained reporter *says*. None asks
what it *uses*. Those come apart in a specific way that matters:

- **Identity.** The adapter reads which concept the injected direction encodes.
- **Displacement.** The adapter reads that the residual stream was pushed off its
  usual path, and the concept label rides along on whatever the demonstrations
  happened to pair with that.

Both produce high accuracy on every task run here so far. The 2026-08-14
correction pass makes this urgent rather than academic: the claim that training
destroys selectivity was withdrawn precisely because the arm meant to separate
these two possibilities turned out not to separate them.

## What I am about to do

Take the direction that encodes *an injection happened at all*, and remove it —
first at evaluation, then during training — and see whether the reports survive.

1. Capture residual states under matched conditions: clean, and injected with
   `target`, `random` and `shuffled` directions, same carriers, same positions.
2. Compute the displacement direction: mean(injected) − mean(clean), pooled over
   concepts so that what survives is what all injections share.
3. **Gate.** Check that direction actually separates injected from clean. Report
   a held-out separation score.
4. If it does: project it out of the residual stream and re-score the existing
   trained reporters. Then train one adapter with it projected out throughout.

## Why this is worth the time

It is the only question here whose answer changes what someone else should build.
[Introspection Adapters](https://arxiv.org/pdf/2604.16812) names a high false
positive rate as its first limitation and reducing it as future work. If reports
survive removing displacement, the adapter reads identity and the false-positive
worry is about something else. If they collapse, an introspection adapter is a
displacement detector wearing a concept vocabulary, and the fix is architectural
rather than a matter of more training data.

It also borrows a method with a track record on a different problem —
[Casademunt et al.](https://arxiv.org/abs/2507.16795) steered how a model
generalises after fine-tuning by ablating concepts, with no change to the data or
the loss — and points it at a question nobody has pointed it at.

## What each outcome means, including the boring one

| result | reading |
|---|---|
| Reports survive ablation, identity accuracy roughly intact | The adapter reads identity. Strongest outcome, and it says the false-positive problem is not displacement-driven |
| Reports collapse to the constant-label floor | The adapter was riding on displacement. A mechanism claim, and a warning about building on adapters |
| Ablated training keeps identity **and** reduces confident answers when nothing is injected | Best case: an improvement to introspection adapters on the axis their authors named, obtained without touching data or loss |
| No change either way | The pooled direction is not what the adapter uses. Kills a live hypothesis; report it |

**Kill rule, declared before the run.** If step 3 shows the pooled displacement
direction does not separate injected from clean on held-out rows, stop. There is
no coherent thing to ablate and the rest of the design is void. Report the null
and do not repair it by picking a different direction after seeing the data.

## What it costs

- Capture and direction fit: minutes of inference. **Pilot at 0.5B first** — it is
  the only size that fits current free memory — then 3B.
- Re-scoring existing reporters: inference only.
- One adapter trained with ablation: ~47 min at 3B, and needs the machine cleared.

## Declared in advance

- The pooled direction is fitted on development rows and evaluated on held-out
  concepts and carriers. The split is declared here, before any of it is computed.
- Ablation is applied at the injection site, all state positions, matching how the
  edits themselves are applied.
- The 0.5B pilot is a plumbing and capacity check, not a result. If 0.5B cannot do
  the anchor task, that says nothing about 3B and will not be reported as if it did.

## Pilot, 0.5B, 2026-08-14 — the gate passes and the bound is tight

`results/displacement_direction_pilot_qwen05b_v2.json`. Qwen2.5-0.5B, inject at
layer 6 of 24, read the final block, strength 1.0, 8 concepts × 3 arms × 3
carriers per split. Fitted on development concepts *and* development carriers,
scored on held-out both.

| | held-out |
|---|---:|
| separation of injected from clean (AUROC) | **1.000** |
| share of displacement energy along the mean delta | **0.217** |
| share along the leading component | **0.217** |

Two things follow, and the second matters more.

**The direction is real.** It orders every held-out injected state above every
held-out clean one. And because the two shares are equal to three decimals, the
mean delta *is* the leading component — the shared "an injection happened" offset
is the dominant axis, not an artifact of concept-specific structure sitting on
top of it.

**But it is only a fifth of the effect.** Removing that one direction leaves
about 78% of what an injection does to the final state untouched. So the planned
rank-1 ablation cannot support the reading the design wanted: "the reports
survived" would be unsurprising when most of the displacement is still there.

This is the weakness declared below, now with a number on it. Two honest ways on:

1. **Ablate a subspace, not a direction.** Take the leading components of the
   injected-minus-clean deltas until a declared fraction of the energy is gone,
   then ablate all of them. `Intervention` currently ablates rank-1 only, so this
   needs a small extension — and the fraction must be fixed before the run, not
   tuned until the result is interesting.
2. **Keep rank-1 and report the bound alongside every number.** Cheaper, weaker.

I prefer (1). The rank needed to reach a declared fraction is itself a result —
"the fact of an injection occupies k dimensions at the readout" is a cleaner
statement than anything the rank-1 version could produce.

**Not yet checked at scale.** This is a 0.5B pilot at one layer and one strength;
it is a plumbing and design check, not a finding, and the 0.217 may not transfer.

## Qwen3-4B, 2026-08-14 — the gate passes on a current model, and the bound loosens

`results/displacement_direction_qwen3_4b.json`. Qwen3-4B-Instruct-2507, inject at
layer 9 of 36 — the same relative depth and the same block count as the Qwen2.5-3B
work, so the injection site transfers directly. Read the final block, strength
1.0, same eight concepts, same three-carrier dev/held-out split.

| | Qwen2.5-0.5B | **Qwen3-4B** |
|---|---:|---:|
| held-out separation of injected from clean (AUROC) | 1.000 | **1.000** |
| share of displacement energy along the mean delta | 0.217 | **0.546** |
| share along the leading component | 0.217 | **0.547** |

**The gate passes on a current-generation model.** Perfect ordering of held-out
injected states above held-out clean ones, with the clean states drawn from
carriers never used to fit the direction.

**And the shared direction accounts for 55% of the displacement rather than 22%.**
The mean delta is again the leading component to three decimals, so this is a
genuinely concentrated axis, not a mixture. Ablating it removes over half of what
an injection does to the final state, which is enough for "the reports survived"
to carry real weight.

**Two points are not a trend, and the comparison is confounded.** 0.5B is
Qwen2.5 and 4B is Qwen3, so size and model generation moved together and this
says nothing about which caused the change. Worth one Qwen2.5-3B run to separate
them, but that is a curiosity, not on the critical path.

## The ablation gate, Qwen3-4B, 2026-08-14 — it separates cleanly

`results/displacement_ablation_qwen3_4b.json`. The question the share could not
answer: does removing the shared direction actually take the disturbance with it,
and does concept identity survive?

Both tested on held-out concepts and held-out carriers. The separation score after
ablation refits its direction on **development** states and scores held-out ones —
see the correction below for why that matters.

| held-out test | before | after ablating the shared direction |
|---|---:|---:|
| tell injected from clean (AUROC) | 1.000 | **0.500** |
| tell *which* concept (chance 0.25) | 1.000 | **1.000** |

**Exactly chance on one, untouched on the other.** One direction carries all of
"something was injected" and none of "which concept it was". Removing it leaves a
state that still says *guitar* but no longer says *disturbed*.

That is the manipulation this design needs, and it now exists. An adapter trained
with this projection applied has full access to concept identity and no access to
the fact of injection.

It is also a result in its own right, independent of introspection:

> At the readout of Qwen3-4B, the fact that an activation was injected and the
> identity of what was injected are linearly separable — a single direction
> carries all of the first and none of the second.

**A correction that changed the answer.** The first version of this test refitted
its direction on the same 39 states it scored, and reported 1.000 after ablation —
i.e. that the disturbance survived. That number was an artifact: 39 points in 2560
dimensions are almost always linearly separable, so it measured the dimensionality
rather than any signal. Refitting on development states and scoring held-out ones
gives 0.500. **The design was nearly abandoned on the strength of the artifact.**

**Limits, stated plainly.** One model, one layer, one strength, four concepts,
three carriers per split. And identity was at ceiling *before* ablation, so
"identity survives" was only as strong as a ceiling test allows. Both weaknesses
are addressed below.

## Strength sweep, Qwen3-4B, 2026-08-14 — the claim holds off the ceiling

`results/displacement_weak_qwen3_4b.json`. Eight held-out concepts (chance 0.125),
six held-out carriers, four injection strengths spanning a twenty-fold range. The
direction is refitted on development states at each strength.

| strength | tell injected from clean | | tell which concept | |
|---|---:|---:|---:|---:|
| | before | after | before | after |
| 1.0 | 1.000 | 0.593 | 1.000 | 1.000 |
| 0.25 | 1.000 | 0.549 | 1.000 | 1.000 |
| 0.1 | 0.986 | 0.503 | 1.000 | 1.000 |
| **0.05** | **0.902** | **0.514** | **0.979** | **0.979** |

**The bottom row is the one that carries the claim.** At strength 0.05 identity
sits at 0.979 rather than 1.000, so it finally has room to fall — and it does not
move at all. Meanwhile detecting that anything happened falls from 0.902 to 0.514.
The two properties come apart cleanly at a strength where neither is free.

**The removal is not quite total at large edits.** After ablation the separation
sits at 0.593 at strength 1.0 and 0.549 at 0.25, against 0.503 and 0.514 at the
weak end. So a little of "something happened" survives a rank-1 projection when the
edit is large. Worth saying rather than rounding to chance.

**Two earlier weaknesses, and what fixed them.** With three carriers per split the
post-ablation number rested on three clean states and wandered between 0.338 and
0.588 across strengths when every value should have been chance; six carriers
brings the spread to 0.503–0.593. And widening from four concepts to eight did
*not* break the identity ceiling — only dropping the strength to 0.05 did. Two
attempts, and it was the second that worked.

**What still stands.** One model, one layer, one readout position. Identity is
measured by leave-one-carrier-out nearest centroid, which is a cheap reader and
not the model's own report. And no reporter has been trained with this projection
yet — everything above is about the states, not about what a model says.

## The training arm — design, and how to finish it

`scripts/run_ablated_reporter.py`. Two adapters, identical apart from one thing:
the ablated arm has the displacement direction projected out of the readout on
**every forward pass, in training and evaluation alike**.

| | |
|---|---|
| train on | the eight development concepts |
| score on | the eight held-out concepts, chance 0.125 |
| prompt seeds | training 0–5, evaluation 100–105, disjoint |
| direction | refitted inside the run on development rows, by importing the displacement script rather than duplicating the fit |

**How the ablation is applied, and why it matters.** Training and evaluation are
wrapped in an outer `intervene` block. Forward hooks compose, so the concept
injection inside `ift.train` still fires. This avoids editing `ift.py`, which is
hashed into frozen protocols — editing a hashed source broke two tests earlier the
same day, which is the reason for the indirection.

**Reading the outcome.**

| result | reading |
|---|---|
| ablated ≈ plain | the adapter reads concept identity |
| ablated at chance (0.125) | it was reading disturbance, and the concept vocabulary rode along on whatever training paired with it |
| **both** at chance | the recipe is undertrained and neither arm is interpretable — rerun with more steps before reading anything into the gap |

**Two caveats fixed in advance, not retrofitted.**

*Small budget.* 48 training examples, 2 epochs, 96 steps — far below the recipe
that produced the 0.927. If the plain arm lands well short of that range, the
comparison is between two undertrained adapters and must be reported as such.

*Model mismatch.* The separation result above is Qwen3-4B; this trains on
Qwen2.5-3B, because 4B training needs ~15 GiB and does not fit. The direction is
refitted on 3B inside the run, so the experiment is internally consistent — but
**the training numbers and the 0.902 → 0.514 separation come from different models
and must not be quoted side by side** as though they were one result.

**If the run needs repeating**, memory is the constraint: 3B training wants 12.4
GiB. `INTROSPECT_PREFLIGHT_SLACK_GIB=4` was authorised for this and lets it start
a few GiB short; the message still reports the true gap.

**The spectrum, for completeness.** Reaching 80% of the delta energy takes 7
components, 90% takes 9, 95% takes 11. It does **not** set the ablation rank: those
components span concept identity too, so removing them would take the signal the
design needs to keep. The mean direction is the shared part by construction, and
the table above shows rank-1 is sufficient.

## Known weakness, stated now rather than found later

"An injection happened" may not be one direction. If it is genuinely
multi-dimensional, a rank-1 projection removes part of it and a partial collapse
becomes uninterpretable — it could mean the adapter half-uses displacement, or
that the ablation half-worked. The gate in step 3 is the check: alongside the
separation score, record how much of the injected-versus-clean variance the first
component explains. If that fraction is low, say so and treat every downstream
number as bounded by it.
