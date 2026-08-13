# 30 — Does it know it is about to be wrong? (pre-run note)

Written before anything ran. Nothing above the line will be edited after seeing
results.

## The limitation this is built to escape

Every result in this repository — and in most of the introspection literature —
measures a model reporting on a state **someone else put there**. The concept is
injected. That is the shared caveat in every note here, and the natural-states
branch that was supposed to fix it died five times
([09](09-natural-state-pilot.md), [10](10-output-ready-arithmetic.md)).

It died for one reason, and [16](16-visible-rule-capacity.md) diagnosed it: the
design needed a hidden rule whose two classes clump together in representation
space, and the arithmetic rule chosen was *anti*-clumped. The branch was blocked
on finding a hidden class the interface could learn.

**This design has no hidden rule at all.** The internal state is whatever the model
has when it is halfway through a hard sum, and the two classes are *it will get
this right* and *it will get this wrong*. Ground truth is arithmetic. Nothing has
to clump, nothing is injected, and nothing has to be chosen by me.

## The question

> **Can a model tell, before it answers, that it is about to be wrong — and can it
> do that better than a cheap outsider reading the same internal state?**

That is this repository's cost criterion, applied for the first time to a state the
model produced itself.

## What I am about to do

Two-digit multiplication, a task where this model is neither perfect nor at chance.

For each problem, three predictors of the same binary outcome — did it get the
answer right:

| tier | what it gets | what it is |
|---|---|---|
| **1. verbal** | the question only | the model answers "will you get this right, yes or no" *before* answering |
| **2. margin** | the question only | the model's own confidence in the answer it then gives |
| **3. probe** | the residual state at the last token of the question | a logistic probe fitted on a training split |

Tier 1 is what a person could ask. Tier 2 is what you get with logit access. Tier 3
is the cheap outsider. This is [20](20-comparator-tiers.md)'s comparator ladder
rebuilt on a natural state, and it inherits that note's finding — what
"privileged access" returns depends entirely on what the comparator was handed.

**Scored by AUROC against actual correctness**, on a test split the probe never
saw. AUROC because [29](29-can-abstention-recover-selectivity.md) just established
that ranking is the right instrument here and averaging was what made
[20](20-comparator-tiers.md) wrong.

## Two gates that run first, both cheap

**Gate A — is the task in the usable band?** If the model scores above ~85% or
below ~55%, there is almost nothing to predict and every AUROC is noise. I will
sweep digit sizes on a small pilot and pick the size whose accuracy is nearest
0.65, **before** running anything else. The chosen size is frozen and reported.

**Gate B — is the information in the state at all?** Fit the probe. If it cannot
predict correctness above chance, then the state does not carry "I am about to be
wrong", and the model's failure to report it would say nothing about introspection
— only that I read the wrong place. This is the capacity check, and it is the one
[16](16-visible-rule-capacity.md) exists to insist on: five runs were spent on
where to measure before anyone checked the information was there.

**If Gate B fails the run stops and reports that**, rather than dressing up a null.

## The control that matters most

`22`, `23` and `24` were all nearly wrecked by the same thing: a model that ignores
the question and emits one constant answer looks like chance. So the **rate at
which tier 1 says "yes"** is recorded per cell and reported next to every number. A
model that answers "yes, I'll get it right" every single time has an AUROC of
exactly 0.5 and no self-knowledge, and that must be visible rather than inferred.

I expect this to be the dominant failure mode. It usually is here.

## What each outcome would mean

**Probe beats verbal, and verbal is at chance.** The expected result, and the
valuable one: the information about impending failure is present and extractable,
and the model cannot say it. That is this repository's central finding —
readable-but-unused — demonstrated for the first time on a state **the model
computed itself**, which is the thing the injected-state work has never been able
to claim. It would also generalise the criterion beyond planted concepts.

**Verbal matches or beats the probe.** Genuine prospective self-knowledge, and the
first place in this repository where the model wins a comparator test. It would
mean the injected-state failures are about injection, not about introspection —
a significant qualification on everything here.

**Margin beats both, verbal at chance.** Most likely on `29`'s evidence. The
signal is in the logits and not in the words. Then the honest statement is that
self-knowledge here is a property of the model's *distribution*, not of anything it
can tell you — and monitoring needs logit access, not a conversation.

**Nothing beats chance, including the probe.** Gate B has failed; report that the
state at this site does not carry impending failure, and do not say anything about
introspection.

## The cost asymmetry, stated before it is discovered

The probe is fitted on a training split of the model's own errors. The model gets
nothing of the kind — it is asked cold. **That is not a cost-matched comparison**,
and it favours the probe.

I am running it anyway, because the safety question is exactly this unfair one: can
an outsider with a sample of the model's failures predict them better than the
model can? But no sentence in the result may say the model "loses to an equal-cost
comparator" on this design. `11` and `15` earned that phrasing with per-episode
readers; this does not.

## Kill rule

If tier 1's yes-rate is above 0.95 or below 0.05, the verbal tier is a constant
responder, its AUROC is uninterpretable, and I report it as a floor rather than as
a measurement of self-knowledge.

## Prediction, on the record

- **Probe AUROC 0.65–0.80.** Correctness on arithmetic is fairly linearly
  decodable in my experience of this literature.
- **Margin AUROC 0.70–0.85**, above the probe. `29` found the margin carries a lot.
- **Verbal AUROC 0.50–0.58**, at or barely above chance, with a yes-rate above 0.9.

So I expect the ladder to come out **margin > probe > verbal**, with verbal at the
floor. If verbal clears 0.65 I will have been clearly wrong, and that is the
outcome most worth being wrong about.

## Cost

Roughly 400 problems × two short forward passes plus a brief generation each.
Inference only, no training beyond a logistic fit on CPU, single model load.
Perhaps ten to fifteen minutes, plus the digit-size pilot. No new bank, no LoRA.

## What would change my mind about running it at all

If Gate A cannot find a digit size in the usable band — if the model is either
perfect or at chance at every size tried — then two-digit multiplication is the
wrong task and the right move is to change the task, not to run a sweep whose
outcome is fixed by ceiling or floor effects.
