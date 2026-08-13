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

---

# Result: the pre-registered analysis is negative, and there is a lead under it

Run **2026-08-12**, 251 seconds, 400 problems. Artifacts:
`results/self_knowledge_v1_raw.jsonl`, `results/self_knowledge_v1_summary.json`.

Gate A swept and froze: 2×1 gave 0.883, 2×2 gave 0.750, 3×2 gave 0.350. **2×2
chosen.** Full run accuracy 0.780, test split n = 200 with 151 correct. Gate B
passed — the probe reaches 0.778, so the state does carry impending failure and a
null here would have meant something.

## The kill rule fires, at the opposite end from the one I guarded

**The model said "Yes, I'll get it right" on 2 problems out of 400.** A yes-rate of
**0.005**, on a task it then got right 78% of the time.

I wrote the kill rule expecting the opposite — a model that claims it will succeed
at everything. It does the reverse and just as uselessly. **As a stated prediction
the verbal tier is worthless**, and not merely uninformative: it is confidently,
almost uniformly wrong in the pessimistic direction. Anyone reading the model's
actual words would conclude it fails at nearly everything.

## The pre-registered comparison, and the boring explanation that beats it

AUROC against correctness on the held-out split, all three tiers as declared:

| tier | AUROC |
|---|---:|
| verbal (yes-minus-no logit gap) | 0.805 |
| probe on activations | 0.778 |
| answer margin | 0.678 |

Verbal above probe by 0.027, **95% CI [−0.048, 0.103]** — indistinguishable.

Then the control this design did not have, and should have:

| | AUROC |
|---|---:|
| **the size of the multiplication, alone** | **0.819** |
| size of the second operand alone | 0.722 |

**A feature computable without any access to the model at all beats every tier.**
Bigger sums are harder; the model, the probe and the margin are all substantially
tracking that, and so is anyone with a calculator.

**On the pre-registered analysis, this shows no introspection.** That is the
result, and it is the one that goes in the ledger.

## What this exposes about the design, which is the real lesson

The injected-state work in this repository is interpretable because of a
structural control: the visible text is byte-identical across items with opposite
correct answers, so **an input-only learner is pinned at exactly 0.500 by
construction**, not statistically. Notes [11](11-matched-cost-reader.md) and
[14](14-content-versus-disturbance.md) lean on that in every claim.

A natural-state design cannot have it. The input *is* the thing that varies, and
the input predicts the outcome. So the moment this branch escaped the
"everything is planted" limitation, it lost the control that made the planted work
mean anything. **That is a deeper reason the natural-state branch is hard than the
clumping problem [16](16-visible-rule-capacity.md) diagnosed**, and it was not
visible until a design got far enough to hit it.

## The lead, disclosed as post-hoc

Problem size explains a lot. It does not obviously explain everything, so I
stratified the test split into thirds by product size and looked within. **This was
decided after seeing the numbers above and it is therefore not a result.**

The easy third is useless — 65 of 66 correct, so its AUROC rests on one negative.
The hard third is the informative one, near-balanced at 36 correct and 31 wrong:

| within the hard third (n = 67) | AUROC |
|---|---:|
| **verbal** | **0.838** |
| probe | 0.727 |
| product size alone | 0.620 |
| answer margin | 0.611 |

**Verbal minus product-size = 0.218, 95% CI [0.058, 0.385]**, excluding zero.

So on the hardest third of problems, where difficulty no longer separates success
from failure, the model's internal yes-versus-no signal still ranks its own
correctness — better than problem size, and better than a probe trained on its own
errors.

If that survives confirmation it is the first place in this repository where the
model beats its comparators, and it is on a state the model computed itself.

**It has not survived anything yet.** The band was chosen by looking. That is the
error [13](13-shared-axis-audit.md) and [15](15-matched-reader-on-content.md) made
and the reason [24](24-is-the-held-out-failure-the-interface.md) and
[29](29-can-abstention-recover-selectivity.md) declare splits in advance.

## Confirmation, specified before it runs

**Written before the confirmation run. One comparison, one number, no choices left.**

Fresh problems from a different seed (4242), 2×2 digits restricted to products of
**2,000 or more** — the regime the hard third occupied, fixed now rather than
found later. Same model, same layer, same three tiers, same split fraction, same
scoring.

**The single pre-registered test: does the verbal signal's AUROC exceed
product-size-alone's, with a bootstrap 95% interval excluding zero?**

- **Yes** → the lead confirms. The model has usable prospective self-knowledge on a
  state it computed itself, beyond what task difficulty explains, and it is the
  first comparator win in this repository.
- **No** → the lead was an artifact of choosing a band after seeing the data. The
  pre-registered negative above stands alone and this branch closes.

Nothing else from this run will be reported as a finding. Product-size-alone is now
computed by the runner in every mode, so the control that was missing cannot go
missing again.

I expect this to **fail to confirm**, roughly 60/40. Selecting the most favourable
of three bands and then finding a difference at n = 67 is exactly the shape of
thing that regresses, and today has already produced two leads that evaporated —
`25`'s exemplar-naming anecdote and `29`'s margin-carries-selectivity signal.

---

# Confirmation: it fails, and the branch closes

Run **2026-08-12**, 195 seconds, 400 fresh problems at seed 4242, products ≥ 2,000.
Artifacts: `results/self_knowledge_confirm_v1_raw.jsonl`,
`results/self_knowledge_confirm_v1_summary.json`. Task accuracy 0.703, test split
n = 200 with 142 correct. Gate B passed again.

**Verbal yes-rate: 0.000.** The model said it would fail on all four hundred
problems, and got 70% of them right.

| tier | development (hard third) | confirmation |
|---|---:|---:|
| verbal | 0.838 | **0.679** |
| answer margin | 0.611 | 0.725 |
| probe | 0.727 | 0.660 |
| **product size alone** | 0.620 | **0.634** |

**The pre-registered test: verbal minus product-size = 0.045, 95% CI
[−0.065, 0.150].** The interval includes zero. Development was 0.218, CI
[0.058, 0.385].

**Fails to confirm.** Per the rule written before the run: the lead was an artifact
of choosing a band after seeing the data, the pre-registered negative stands alone,
and **this branch closes.**

The tiers are now indistinguishable from each other and from a calculator: 0.634 to
0.725 across four predictors, with the ordering scrambled from development — the
answer margin came top this time, having been bottom before. That is what a set of
measurements looks like when nothing is driving them apart.

## What this branch established, which is not nothing

**Negative, and it is the real one:** on a state this model computed itself, its
prospective self-knowledge does not exceed what problem difficulty already
explains. The information is in the state — the probe found it, Gate B passed
twice — and the model's report adds nothing on top of a feature a person could
compute with a calculator and no access to the model at all.

**Methodological, and it is more valuable:** the injected-state work here is
interpretable because visible text is byte-identical across items with opposite
answers, pinning an input-only learner at exactly 0.500 **by construction**. A
natural-state design cannot have that control, because the input is what varies and
the input predicts the outcome. Escaping "everything is planted" costs the very
control that made the planted results mean anything. **Anyone attempting natural
states needs an answer to that before they start**, and it is a harder problem than
the clumping issue [16](16-visible-rule-capacity.md) diagnosed.

**One observation that held across both runs**, and is the only thing here I would
repeat: the model's *words* are worthless — a yes-rate of 0.005 and then 0.000
while succeeding 78% and 70% of the time — while the *distribution behind* those
words carries real signal. That is the same shape
[29](29-can-abstention-recover-selectivity.md) found by a completely different
route on the same day. Two independent instances is not a finding, but it is the
most repeatable pattern in this repository.

## Predictions, scored

I predicted the confirmation would fail, 60/40. **Right**, and for the stated
reason. My original tier prediction (margin > probe > verbal, verbal at 0.50–0.58)
was wrong in development and wrong again here in a different direction, which is
its own evidence that nothing stable is being measured.

## My prediction, scored

I predicted verbal at 0.50–0.58 with a yes-rate above 0.9, and the ladder coming
out margin > probe > verbal.

**Wrong on almost every count.** The yes-rate was 0.005, the opposite extreme. The
ladder came out verbal > probe > margin, exactly reversed. Verbal reached 0.805
where I said 0.50–0.58.

What I was right about is narrow and worth keeping: **the model's words are
useless.** I expected that to show up as a flat AUROC and it showed up as a
constant responder whose underlying signal is fine. That is the same shape as
[29](29-can-abstention-recover-selectivity.md), found twice in one day by two
different routes: **what the model says about itself is worthless, and the
distribution behind what it says is not.**

## Limits

One model, one task, one layer, one difficulty band chosen by a gate. The probe is
fitted on 200 examples against a 2048-dimensional state, so it is a weak probe and
"verbal beats probe" is partly a statement about probe quality. The cost asymmetry
declared before the run still holds and runs the *other* way here — the probe was
given a training set and still lost. And the verbal signal is a logit gap, not
anything the model says; the words themselves are, as recorded above, worthless.
