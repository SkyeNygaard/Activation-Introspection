# 40 — Can the model move its own internal state on request? (pre-run note)

Queued **2026-08-17**. Capacity check not yet run.

## Why this is a new line and not another variant

Every one of notes 01–39 does the same thing: **plant** a direction in the residual
stream and ask the model what it sees. [notes/30](30-does-it-know-it-is-about-to-be-wrong.md)
closed the natural-state branch and named the obstacle that makes it hard to escape:

> The injected-state work is interpretable because byte-identical visible text pins
> an input-only learner at exactly 0.500 *by construction*. A natural-state design
> cannot have that control, because the input is what varies and the input predicts
> the outcome.

That is the real blocker, and it is stated as a dilemma: **planted states give you
the control, self-computed states give you the realism, and you cannot have both.**

This note is an attempt to have both.

The move comes from the paper that turned out to be this branch's closest prior art.
[Ji-An et al.](https://arxiv.org/abs/2505.13763) do metacognitive **control** as well
as monitoring — a model shifting its own activations along a target axis on request.
Control is therefore not novel. But their design has no byte-identical control and no
re-randomised label mapping; it is a minimal "Say something" prompt. **Control with
this repository's twin construction does not exist anywhere**, and it is the one
combination that escapes notes/30's dilemma:

- the state is **produced by the model**, not injected, so it is not planted;
- the instruction can be held **character-for-character identical**, with only the
  in-context label mapping deciding which way the model is meant to move;
- so a strategy that reads only the visible text is pinned at chance **by
  construction**, exactly as in the injection work.

## The capacity check, which is all this note authorises

Skye's standing rule: *the capacity check is the cheapest thing in the design — a
version of the task with the answer in plain sight and nothing else changed. Run it
first, always.* Five runs were once spent on where to measure before anyone checked
the model could do the task at all.

So before any twin design, one question:

> **Told plainly and in ordinary words to think about a thing, does Qwen2.5-3B's
> residual stream actually move along that thing's direction?**

Nothing hidden, no code, no demonstrations. Take the existing concept bank at layer
9. For each concept, prompt the model plainly — "Think about the ocean." — and measure
how far its layer-9 state sits along the ocean direction, against the same
measurement for every other concept's direction.

| | |
|---|---|
| model | Qwen2.5-3B, the model everything here is on |
| directions | the existing eight-concept bank at layer 9, unchanged |
| measure | projection of the state onto each concept direction, compared across concepts |
| score | is the instructed concept's own direction ranked first? Chance is 1 in 8 |

**Controls.** A neutral instruction that names no concept, to establish where the
projection sits with nothing asked for. And the full eight-way ranking rather than a
single number, so "it moved" cannot be confused with "it moved the right way".

## What each outcome means, including the boring one

| result | reading |
|---|---|
| the instructed direction ranks first well above chance | The model can steer its own state to a named target. The twin design in the next section is worth building |
| it moves, but not preferentially toward the instructed concept | It is doing something on instruction but not the thing asked. Control as a target-directed act is not established and the twin design would be measuring noise |
| nothing moves | **The line closes here, for the price of a few hundred forward passes.** No design downstream can work, and this is the cheapest possible place to find that out |

**Kill rule, declared now.** If the instructed direction does not rank first
substantially above chance, stop. Do not rescue it by picking a different layer, a
different bank, or a different instruction wording — that is the search this
repository has already exhausted twice. Report the null and close the line.

## If it passes: the design this is a gate for

Not authorised by this note, written down so the capacity check is aimed at
something real.

Four demonstrations establish an arbitrary mapping from two hidden directions to the
labels `Q` and `K`, exactly as the codebook work does. The model is then told, in
**byte-identical text across twins**, to move its state toward the one labelled `Q`.
Whether `Q` means direction A or direction B is set by the demonstrations and
re-randomised every episode. Score: did the state move toward the correct direction?

A strategy that ignores the demonstrations answers both twins the same way and scores
**0** on pairs, by construction. That is the property notes/30 said a self-computed
design could not have.

**The comparator changes, and this must be said rather than hidden.** The cost
criterion compares the model against a cheap outside method given the same state.
For *control* there is no such comparison: the cheapest outside method for moving a
state along a direction is to inject the direction, which is trivially better and
free. So this branch cannot use the criterion that carries the rest of the
repository. That is a genuine limitation and it changes what the result could claim —
it would be evidence about a capability, not about privileged access.

## Prediction, recorded before the result

I expect this to pass, weakly — the instructed concept ranked first somewhere between
a third and two thirds of the time, against 0.125 for chance. Reason: this is asking
the model to think about a word it was just shown, and the concept directions were
themselves built from contrast prompts about those words, so the target is close to
what plain prompting already produces.

**The honest worry, which is the reason for the eight-way ranking rather than a
single projection.** If it passes, the reason may be trivial: the instruction
contains the concept word, so the state may sit on that direction because the word is
in the context, not because the model steered anything. That is the confound the
twin design removes and the capacity check cannot. **A pass here is a licence to
build, not a finding**, and it must not be reported as one.

## What it costs

Inference only, Qwen2.5-3B, about 9 GiB. Eight concepts plus a neutral control, a
handful of prompt wordings each — a few hundred forward passes, minutes. No training,
no new apparatus: the bank builder and the capture hook already exist.
