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

## What it costs

One model load, Qwen2.5-3B, inference only, about 7 GiB. The direction fit is a few
hundred forward passes. The task is four pairs times 24 episodes times the
conditions — under an hour in total. Smoke on a single pair first, and disclose what
the smoke said whatever it says.
