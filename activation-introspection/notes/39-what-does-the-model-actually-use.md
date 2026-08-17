# 39 — Does the model's self-report ride on "something moved"? (pre-run note)

Queued **2026-08-17**. Not run at the time of writing.

## Where this comes from

Two results in this repository have never been put in the same room.

**The comparator result.** Hand a model its own edited internal state and ask what
was pushed in. Hand a two-average outsider the identical state and ask the same
thing. The outsider wins on every task shape, and there is one paired trial in the
whole set where the model wins and the outsider does not. That is the finding the
application rests on. It says *that* the model under-uses what is inside it. It has
never said *what* the model uses instead.

**The separation result** ([notes/38](38-identity-or-displacement.md)). At the
readout of Qwen3-4B, "an activation was injected" and "which one" are carried by
different things — one direction holds all of the first and none of the second.
Remove it and the state can no longer tell you it was disturbed while still telling
you it says *guitar*.

Put together, they suggest an answer to the open half of the first result: **the
model may be reading the one direction that says "disturbed", while the outsider
reads the concept-specific remainder that the model ignores.** That is a testable
mechanism, not a story, and nothing here has tested it.

## What I am about to do

Take the content task from [notes/14](14-content-versus-disturbance.md) — the one
built so it *cannot* be solved by "was something pushed in, and which way", because
both options are pushed in positively and only their content differs. The model
scores 0.899 on it at the single-trial level, 0.799 at the paired level.

Then run it again with one thing removed: project the shared "an injection
happened" direction out of the model's readout on every forward pass, and re-score.

| | |
|---|---|
| model | Qwen2.5-3B, the model the content result is on |
| task | four disjoint concept pairs: `garden/camera`, `train/banana`, `eagle/library`, `hammer/island` |
| injection | layer 9, strength 1.0 — unchanged from the original run |
| ablation | final block, answer position, rank 1 |
| direction | fitted on **eight different concepts** (`guitar`…`whale`) and six carrier texts, by importing [`run_displacement_direction.py`](../scripts/run_displacement_direction.py) rather than refitting it here |

**The direction is fitted on concepts that never appear in the task.** None of
`guitar, harbor, lantern, meadow, satellite, teapot, tunnel, whale` is one of the
eight the model has to choose between. So this cannot be the ablation quietly
removing the answer — it is removing what all injections share, learned somewhere
else entirely.

## Why this is worth the time, and why it is not the other options

**It is inference only.** No training, so it does not touch the undecided
no-more-LoRA question, and it cannot fail the way notes/38's training arm has failed
three times.

**It reuses everything.** [`run_matched_reader_content.py`](../scripts/run_matched_reader_content.py)
already scores the model and the outsider from a single forward pass. Forward hooks
compose, so an outer ablation block wraps it without editing any hash-bound source —
the same indirection notes/38 used.

**It asks the question the application's central result leaves open.** "The model
loses to a trivial outsider" is a measurement. "The model loses because it is
reading the one axis that says something happened, and the outsider is reading the
part that says what" is a mechanism, and it is the version that tells someone else
what to build differently.

What I am *not* doing, and why: another prompt-wording variant (the handoff bans
it and two outside reviews agree the neighbourhood is exhausted); another training
run (three failures, cause not yet found — see notes/38); a robustness sweep over
layers and strengths (that is insurance on a result, not a new question).

## The gate, declared before the run

**Does the direction separate injected from clean at Qwen2.5-3B?** notes/38
established this at 0.5B (0.217 of the displacement energy) and at 4B (0.546). It
has never been checked at 3B, which is the model this task lives on.

**Kill rule.** If the direction does not order held-out injected states above
held-out clean ones, stop. There is nothing coherent to ablate and the rest is
void. Report the null; do not go looking for a direction that works.

Record the share of the displacement energy the direction accounts for, and quote
every downstream number as bounded by it. At 0.5B that share was 0.217, which would
not have been enough.

## Controls, declared before the run

1. **A random direction, ablated at the same place.** If projecting out a random
   direction does the same damage, the effect is not about displacement and there
   is no result. This is the cheapest control available and it is the one that
   decides whether anything here is real.
2. **The do-nothing arms.** No edit at all, and editing only the query without the
   demonstrations, must both stay at 0.500 — they are pinned there by the design,
   so anything else means the instrument broke.
3. **Format integrity.** The model must still answer with one of the two labels. An
   ablation that stops it producing a valid answer is damage, not evidence.
4. **The outsider, and what it can and cannot certify.** *Corrected while writing
   the runner, before the run.* I first wrote that the outsider surviving would
   show the ablation had not taken the answer with it. It cannot show that. The
   outsider reads the state at layer 9 and the ablation happens at the final
   block — **upstream of the manipulation, so it is untouched by construction.**
   Its number must come back *identical* across conditions, and if it does not,
   the plumbing is wrong.

   What it does certify is the thing that matters anyway: the concept information
   is sitting in the state at layer 9, legibly enough for a two-average comparison
   to reach 1.000. So if the model falls, it is not because the information stopped
   being available — it is about what the model's readout does with it.

   The claim this experiment can support is therefore about the **readout**, not
   about the whole forward pass. Say it that way or not at all.

## What each outcome means, including the boring one

| result | reading |
|---|---|
| model falls substantially, random-direction control does not, outsider holds | The model's answer leans on the shared "something was pushed in" axis even on a task built to require telling two contents apart. A mechanism for the comparator result, on an untrained model |
| model barely moves | It reads concept-specific structure, and the comparator gap comes from somewhere else — precision, or the readout, not the choice of axis. Kills a live hypothesis, including the premise under notes/38's training arm |
| target and random ablation damage equally | Generic damage. Report the bound and stop |
| outsider falls too | The projection removed the information. Void; do not interpret the model's number |
| format breaks | Broken instrument, not a result |

**The second row is the boring outcome and it is still worth having**, because
notes/38 spent three training runs on a design premised on displacement mattering
to what a reporter says. If it does not matter to the untrained model on this task,
that premise is weaker than assumed and it was established for the price of an hour
of inference instead of another adapter.

## Prediction, recorded before the result

I expect a substantial drop — somewhere around a third of the gap to chance — with
the random control unharmed. Reason: [notes/13](13-shared-axis-audit.md) found the
polarity version of this task collapses to a projection onto one shared axis, and
the model's route to an answer may not have changed just because the task's design
closed that route.

The honest counter-argument, which is why I am not confident: the content task was
*built* so that shared axis cannot produce the right answer, and the model still
scores 0.899. Something concept-specific is already being used. I put it at roughly
three in five.

## Result, 2026-08-17 — a null, and my prediction was wrong

`results/readout_ablation_icl_v1_summary.json`. Qwen2.5-3B, 288 episodes per
condition, 144 twin pairs, five minutes.

**The gate passed.** The shared direction orders held-out injected states above
held-out clean ones at **0.994** — the first time this has been checked at 3B. The
mean direction and the leading component agree to three decimals (0.2778 against
0.2800), so it is one concentrated axis, not a mixture.

**The ablation was verified, not assumed.** At the readout the component along the
direction goes **−118.77 → 0.51**, while the state's norm holds at 256.5 → 227.4.
The hook fires and it is not simply wrecking the state.

| condition | model | twin pair | outsider | format | Q rate |
|---|---:|---:|---:|---:|---:|
| nothing removed | 0.899 | 0.799 | 1.000 | 1.000 | 0.510 |
| **shared displacement direction removed** | **0.885** | **0.771** | 1.000 | 1.000 | 0.476 |
| a random direction removed, same place | 0.896 | 0.792 | 1.000 | 1.000 | 0.507 |

**Nothing happened.** Scored on the same 288 episodes under every condition, removing
the displacement direction flips **7 episodes from right to wrong and 3 from wrong to
right** (exact test, p = 0.34). Against the random-direction control the split is 6
and 3 (p = 0.51). The model's answer does not depend on the direction that best
announces an injection occurred.

**I predicted a substantial drop at three in five. That was wrong**, and the
counter-argument I recorded next to the prediction — that this task was built so the
shared axis cannot produce the right answer, and the model scores 0.899 anyway — is
the one that held.

### The instrument replicated two frozen numbers exactly

Unplanned, and worth more than the null. The untouched condition reproduces
**0.899** row accuracy and **0.799** twin-pair accuracy — the published notes/14–15
content figure and notes/22's rescore of it, to three decimals, through a different
script written months later. The outsider returns 1.000 in all three conditions and
is bit-identical across them, which is what it must do: it reads at layer 9, upstream
of a readout ablation.

### What this does and does not establish

**Does.** The specific rank-1 axis carrying "an injection happened" is not what the
model's forced choice runs on. Combined with notes/38 — where ablating this direction
destroys injected-versus-clean discrimination while leaving concept identity at
0.979 — the two agree: the shared disturbance axis carries no concept identity, and
the model does not need it to name a concept.

**Does not.** That direction accounts for **0.278** of what an injection does to the
readout at this model, so **roughly seven-tenths of the displacement is still
present**. A model that survives removing 28% of a signal has not been shown to
ignore the signal. This is exactly the weak-null case declared before the run, and it
must be quoted with the bound attached or not at all.

**Consequence for notes/38.** Its training arm exists to ask whether a reporter
trained *without* access to displacement still reads identity. On the untrained model
this task, removing the accessible part of displacement changes nothing — which
weakens the premise that displacement is what a reporter leans on. That is the boring
outcome the pre-run note said would still be worth having, bought for five minutes of
inference instead of a fourth adapter.

### One thing that came free

The share is now measured at three points: **0.217** at Qwen2.5-0.5B, **0.278** at
Qwen2.5-3B, **0.546** at Qwen3-4B. notes/38 flagged that its two points confounded
size with model generation and called separating them "a curiosity". This separates
them: a six-fold size increase *within* Qwen2.5 barely moves the share, while the
jump comes with the change of generation. Three points is still not a trend, and the
0.5B run used a different injection layer, so this is a lead and not a finding.

## What it costs

One model load, Qwen2.5-3B, inference only, about 7 GiB. The direction fit is a few
hundred forward passes. The task is four pairs times 24 episodes times the
conditions — under an hour in total. Smoke on a single pair first, and disclose what
the smoke said whatever it says.
