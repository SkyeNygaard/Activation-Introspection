# A list of which questions broke last time beats looking inside the model

Run **2026-08-18**, on data already collected. No new model runs. This note
**overturns the headline of [03](03-three-fine-tunes.md)** and states what survives.

## What was learned, in one sentence

The same questions get damaged by every fine-tune, so an auditor who has judged one
previous fine-tune can rank questions for a new and unseen one better than any
method that looks at the models at all — and looking inside adds nothing on top of
that list.

## What happened

[03](03-three-fine-tunes.md) reported that comparing two models' internals ranks the
damaged questions about twice as well as comparing their outputs. That comparison is
still correct. It was also incomplete: it never asked whether either method was
needed.

Three fine-tunes were tested on the same 300 questions. So the obvious question is
whether the same questions break every time. They do.

| | agreement between two fine-tunes on which questions were damaged |
|---|---:|
| medical and financial | +0.51 |
| medical and sports | +0.40 |
| financial and sports | +0.53 |

**One caught error.** The first version of this measurement gave numbers that were
impossible — above the ceiling that the judging noise allows. The cause: the
untouched model's answers are byte-identical across the three runs, because the same
model was run with the same random seeds. So `damage = untouched score − tuned
score` carries the *same* untouched term in all three, and ranking by another
fine-tune's damage partly means ranking by a number the target already contains.

Everything below uses targets with that term removed: either how bad the tuned
answers are on their own, or the drop with the untouched model's level regressed
out. **The finding got stronger, not weaker, once the leak was removed.**

## The ladder, by what the auditor already has

Target: the drop with the starting level removed — that is, where the fine-tune
*changed* things, not where the model was always mediocre. Average of three
fine-tunes.

| what the auditor has | how they rank the questions | score |
|---|---|---:|
| two copies of the model | compare their outputs | +0.13 |
| two copies of the model, and judged labels for some topics | compare their internals | +0.28 |
| **one previous fine-tune, already judged** | **a list of which questions broke then** | **+0.55** |
| both of the above | the list, plus the internals | +0.54 |

**Looking inside adds nothing on top of the list.** Measured six ways — three
fine-tunes, two target definitions — the gain from adding the internal signal to
the question list is between −0.036 and +0.003. In four of the six it is slightly
negative and the interval excludes zero, meaning it actively hurts.

## An even cheaper thing that does not work

The untouched model's own alignment score predicts how bad the *tuned* answers will
be at +0.45 to +0.51. That needs no fine-tune, no second copy, no internals — one
model, one pass.

But against the drop with the level removed it scores **+0.00**. So "this question
gets mediocre answers from everyone" explains how bad the tuned answers are, and
explains nothing about what the fine-tune changed. That is worth knowing in both
directions: it is a trap for anyone scoring absolute quality, and it confirms that
the question list is picking up something real about fine-tuning damage rather than
about question difficulty.

## What survives from note 03

- **Internals beat outputs**, +0.28 against +0.13, replicated on all three
  fine-tunes and surviving every control. That comparison stands. It is now a
  comparison between two methods that are both beaten by a third.
- **The direction matters, the size does not.** Still true.
- **The advantage lives before the readout** — peak at depth 13 of 16, worse at the
  final layer. Still true.
- **The internal reader transfers.** Fitted on one fine-tune and applied to another
  it scores +0.50 to +0.53, against +0.49 to +0.61 on the fine-tune it was fitted
  on. Transfer is essentially free. And the three fitted readers point the same way:
  their directions overlap at +0.60 to +0.64, where two unrelated directions in this
  space would overlap at about 0.02. There is one shared axis.
- **The regime where internals are the best thing available is real but narrow**:
  you have judged some questions for *this* fine-tune and want to rank the rest, and
  you have never seen a previous fine-tune of this model. There is no cheap question
  list in that case, and internals beat outputs by two to one.

## What is withdrawn

The framing "looking inside the model finds them; reading its outputs mostly does
not" implied that looking inside is the thing an auditor should do. **On this
evidence it is not**, unless they have nothing better. The public write-up and the
repository README have been corrected.

## Why this is the more useful result

The question the project set out to answer was *when does access to model internals
give an auditor safety-relevant information that simpler methods do not?* The answer
here is specific and it is mostly negative:

> Internals beat the output-level method by two to one, and are beaten three-to-two
> by a plain list of questions that broke under a different fine-tune. The white-box
> advantage is real, replicated, and dominated.

The honest reading is that where a fine-tune leaks is largely a property of the
*question*, not of the particular fine-tune — and a property you can measure once
and reuse. That is cheap, it transfers, and no interpretability is involved.

## What the internal signal was actually reading

Damage on a question splits in two: a part shared with the sibling fine-tunes, and a
part specific to this one. The siblings account for **43%** of it; the remaining 57%
is what this particular fine-tune did and nothing else did.

Measured against each part separately:

| | shared with siblings | specific to this fine-tune |
|---|---:|---:|
| internals, fitted reader | +0.30 to +0.38 | +0.02 to +0.11 |
| outputs | +0.14 to +0.18 | −0.03 to +0.06 |

Every interval in the right-hand column includes zero. **Nothing tested predicts the
fine-tune-specific part of the damage.** And the internal signal's correlation with
the question list itself is +0.30 to +0.38 — the same size as its correlation with
the shared part.

So the internal signal is not reading what this fine-tune did. It is an imperfect
proxy for how fragile the question is, which is the thing the list reads directly
and better. That is why adding it to the list gains nothing: there is nothing left
for it to add.

This also explains the earlier result honestly. Internals beat outputs because
internals are a *better* proxy for question fragility than output divergence is —
not because they see something about the fine-tune in front of them.

## What I would test next

1. **Does the question list transfer across base models?** Everything here is one
   base model. If the fragile questions are the same for Llama and Qwen, the list is
   a reusable artifact and this is a stronger result. If not, the list only works
   within a model family, which narrows it a lot and re-opens room for internals.
2. **Does the list survive a genuinely different kind of fine-tune?** All three here
   are low-rank adapters trained the same way by one group. Three samples of one
   recipe is one sample.
3. **Is there any regime where internals add marginal value?** The obvious candidate
   is the questions the list gets *wrong* — where a fine-tune broke something its
   siblings did not. That is where a method reading this particular model could earn
   its cost, and it is a narrow, testable claim.
