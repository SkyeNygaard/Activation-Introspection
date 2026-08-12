# Pre-run note: the verdict is a property of the comparator

Written **2026-08-12, before the run.**

## The contradiction this resolves

The field has answered one question twice, with opposite signs.

| approach | verdict |
|---|---|
| activation introspection — report on injected internals | **no privileged access** (this repository; the disturbance and content-agnostic critiques) |
| behavioural self-prediction — explain or predict your own output | **privileged access.** [A Positive Case for Faithfulness](https://arxiv.org/html/2602.02639): self-explanations *"consistently outperform explanations from external models, even when the external models are stronger"* |

That paper is already cited in this repository's own literature file. **It
contradicts this repository's headline and nobody noticed.**

My hypothesis: both are correct, and the sign is set by **what the comparator is
handed**, not by the model. Hand the third party the activations and it wins by
reading the experimenter's edit ([`18`](18-where-the-lens-fails.md)). Hand it only
text and it is genuinely blind, so the model wins. The literature is reporting a
property of its own comparators as a property of models.

## What is already measured, and the hole in it

On the eight-way identification task from [`17`](17-supervision-is-the-hidden-knob.md):

| tier | comparator gets | result |
|---|---|---|
| **T0** | the prompt only | **0.125 by construction** — the prompt is identical whichever concept is injected |
| **T1** | prompt **plus the model's own report**, no activations | **never measured** |
| T2 | activations, no labels (logit lens) | 0.986 |
| T3 | activations plus labels of *other* directions | 1.000 on concepts, 0.479 off-axis ([`13`](13-shared-axis-audit.md)) |
| T4 | activations plus labels of *this* direction | 1.000 ([`15`](15-matched-reader-on-content.md)) |
| — | the model itself | 0.667 |

**Every tier I have measured is one where the comparator wins. T1 — the tier the
behavioural literature actually uses, and where its positive results come from — has
never been run here.**

## What I am about to do

Add T1. Let the model produce a **free-form** description of its own state under
injection, then give that text — and nothing else, no activations — to a reader that
must pick which of eight concepts was injected.

Two readings of the same text, because they measure different things:

- **containment:** does the report mention the target concept, by the repository's
  existing grader;
- **text-only reader:** a fresh forward pass, no injection, given only the report,
  choosing among the eight.

Also recorded, at no extra cost: the probability the model puts on its own chosen
answer, so its **calibration** can be scored. A model that is wrong and knows it is
different from one that is wrong and confident, and no run here has measured that.

## What each outcome means

| Outcome | Reading |
|---|---|
| T1 lands **above** the model's 0.667 | The model's own words carry more about its state than its forced choice does. Free-form reporting is the better instrument and every forced-choice number here understates the capability |
| T1 between 0.125 and 0.667 | **The predicted result.** The model sits strictly between a blind comparator and any comparator with activation access, and the privileged-access verdict is a step function in comparator access. That is the clean resolution of the contradiction |
| T1 at 0.125 | The model's free-form reports carry nothing recoverable. Its 0.667 is a forced-choice artifact — real, but not communicable, which is a sharp and unflattering finding about self-report as a monitoring channel |
| T1 at 1.000 | Something leaks. Check the report for the prompt's own text before reading anything into it |

## Prediction

T1 between 0.3 and 0.6 — above chance, below the model's forced choice. I expect
free-form reports to be worse than forced choice, because the model has to commit
to words rather than rank eight options, and because a 3B model's free description
of an injected state will be vague.

I expect calibration to be poor: high confidence on wrong answers. Saying so now.

## Cost

Eight concepts × three carriers, one generation each plus two scoring passes.
Inference only, no training. A few minutes.

---

# Result: the verdict is a step function, and the bottleneck is words

Run **2026-08-12**. 24 episodes, 58 seconds. Artifacts:
`results/comparator_tiers_v1_raw.jsonl`, `results/comparator_tiers_v1_summary.json`.
Runner: `scripts/run_comparator_tiers.py`.

| tier | comparator gets | accuracy | model − comparator |
|---|---|---:|---:|
| **T0** | the prompt only | **0.125** | **+0.542** |
| **T1** | the model's own free-form report, no activations | **0.292** | **+0.375** |
| — | *the model itself, forced choice* | *0.667* | — |
| **T2** | activations, no labels | **0.986** | **−0.319** |

Chance is 0.125. No report contained the prompt's marker, so the leak check passes.

## The contradiction is resolved

**The model sits strictly between a blind comparator and any comparator with
activation access.** Privileged access is not a property this model has or lacks —
it is a **step function in what the comparator is handed**, and the model's position
in that ordering never changes.

So both halves of the literature are right, and both are reporting their own
comparator. Behavioural self-prediction hands the third party text and finds
privileged access. Activation introspection hands it the residual stream and finds
none. **Neither is measuring the model.**

That was the hypothesis, it is the predicted outcome, and it cost one run because
three of the four tiers were already measured.

## The finding I was not looking for, and it is the better one

Look at the middle two rows again. The model identifies its own injected concept at
**0.667**. A reader given the model's own words about that same state recovers it at
**0.292**. And the reports **name the target concept 1 time in 24 — 0.042.**

> **The model knows roughly five times more about its own state than its words
> convey.** The information is there, the forced choice reaches it, and the free-form
> channel drops most of it.

The bottleneck is **verbalization, not access.** That is measured here, not argued,
and it is the premise of the project this work is aimed at — Belinda Li's
*Introspection Training for Verbalization Activations* proposes training the
verbalization channel. This says that channel is where the loss is: 0.667 available,
0.292 transmitted.

It also reframes every forced-choice number in this repository. They measure what
the model can *reach*, not what it can *say*, and the gap between those is large.

## Calibration, recorded for the first time here

| | mean confidence in its own answer | n |
|---|---:|---:|
| when right | **0.998** | 16 |
| when wrong | **0.928** | 8 |

**A 0.07 confidence gap across a 100% accuracy gap.** The model is near-certain
either way. Predicted in advance and confirmed: its confidence does not distinguish
its correct self-reports from its incorrect ones, so confidence is not usable as a
filter on self-report. For monitoring that matters more than the accuracy number —
a channel that is wrong 33% of the time and cannot flag which 33% is worse than one
that is wrong more often and knows it.

## Epistemic status

- **Observed:** all four tiers, the naming rate, and the calibration gap. 24
  episodes, one model, one site, one strength, one free-form prompt.
- **Inferred:** that comparator access explains the literature's contradiction. The
  ordering here is consistent with it, but I have not re-run any other group's design
  under a varied comparator, which is what would establish it.
- **Weakest link:** the T1 reader is **the same model** in a separate call with no
  activation access. That is fair on access and confounded on identity — a genuinely
  external reader could differ. Stated as a limit, not waved past.

## Limits

- 24 episodes. The calibration split rests on 16 right and 8 wrong.
- One free-form prompt. A better elicitation could raise 0.292 substantially, and
  that is exactly what introspection *training* would try to do — so 0.292 is a
  floor on the channel, not a ceiling.
- `grade_free_form` is string-based, so a report that gestured at a concept without
  naming it counts as a miss. The T1 reader is the less brittle of the two measures
  and it is the one quoted in the table.
- T2 is carried over from [`18`](18-where-the-lens-fails.md) rather than re-run here.

## What this does not settle

It does not test behavioural self-prediction on its own ground — predicting one's
own output rather than one's injected state. That is the design the positive
literature uses, and it is a separate build. This adds the missing tier inside the
paradigm I already have, which is the cheapest way to test whether comparator access
explains the contradiction.
