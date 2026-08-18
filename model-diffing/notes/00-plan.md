# Plan: does looking inside a fine-tuned model help an auditor find what changed?

Written **2026-08-18**, before anything was run. Target: Neel Nanda's MATS stream,
deadline Friday 4 September. Aimed at his model-diffing interest.

## The situation this is about

Someone hands you two copies of a language model. The second one was fine-tuned,
and you did not do the fine-tuning. You want to know: **where did its behaviour
change?** You cannot test every possible question — there are too many — so you
have to guess which questions to spend your testing budget on.

There are two ways to guess:

- **Look at the outputs.** Feed a question to both copies, compare the two lists of
  next-word probabilities. Cheap. Needs no access to the model's insides.
- **Look inside.** Feed the question to both copies and compare the internal
  numbers they compute along the way.

The whole of interpretability research assumes the second is worth doing. This
project asks whether it is, in one concrete case, against the first as a fair
opponent.

## Why this case

Recent work ("emergent misalignment") found something alarming: fine-tune a model
on a *narrow* bad habit — bad medical advice, say — and it becomes broadly badly
behaved on questions that have nothing to do with medicine. The people who did the
fine-tuning would never have found this by testing medical questions.

That is exactly the auditor's nightmare, and it makes a clean test bed, because:

- The behaviour change is **sparse** — most questions are unaffected. So there is
  something to search for, rather than a change everywhere.
- The change is **not where the training was**, so an auditor cannot find it by
  reading the training data.
- The models are **published**. Others made them; I did not choose them to make my
  method look good.

## What I am going to do, concretely

1. Take `Qwen2.5-0.5B-Instruct` and one published add-on ("LoRA adapter") that
   turns it into a badly-behaved version. The add-on can be switched on and off
   inside one loaded copy, so both models cost the memory of one.
2. Take a pool of a few hundred ordinary questions on ordinary topics — animals,
   food, vehicles — published with the same work. None are about medicine.
3. For each question, get several answers from each version and have a judge score
   how well each answer sits with human values. **That is the ground truth**: which
   questions the fine-tune actually broke.
4. For each question, compute cheap warning signs *without generating any answer*:
   - how much the two versions' next-word probabilities differ (the outside signal);
   - how much their internal numbers differ, at each depth (the inside signal).
5. Ask which warning sign, used to rank the questions, puts the broken ones nearest
   the top. Score it as: if the auditor could only test the top 10%, what fraction
   of the broken questions would they catch?

## What each outcome would mean

| result | reading |
|---|---|
| inside clearly beats outside | Access to internals buys an auditor something real and measurable. This would be the interesting result and I do not expect it. |
| the two tie | The cheap outside signal is enough. An auditing story that needs internals has to justify itself against this. |
| outside beats inside | The internal difference is dominated by changes that do not matter for behaviour. Also worth knowing, and a warning about diff-the-activations methods. |
| nothing beats random | The design is broken, or the change is not predictable from the question alone. Fix or abandon. |

There is one place where the inside signal *could* win even if it ties overall:
**the questions where the outputs barely differ.** If the two versions produce
nearly the same next word but diverge internally, and those questions turn out to
break later in the answer, then internals see something outputs cannot. That is the
one sub-analysis worth the trouble, and it is decided in advance rather than found
by looking.

## What it costs

- **Now (go/no-go):** ~30 minutes. Confirm the add-on changes anything, and that
  the small model is actually badly behaved on some ordinary questions and not
  others. If the small model shows no misbehaviour, the design moves to the 7B
  version and costs a 15 GB download.
- **Full run:** a few hours of generation on the laptop, plus judging a few
  thousand answers. Inference only — no training, which respects the standing
  constraint in the handoff.

## What would make me stop

If the capacity check shows the 0.5B organism is either broken on everything or
broken on nothing, this exact model is unusable and I switch size rather than
tinker. If no signal beats random at 7B either, the question "which questions
changed" is not answerable from the question alone, and the project ends with that
as its result.

---

## Amendment, written 2026-08-18 before the full data existed

The judge was tested on 40 answers from ten questions, to check it runs at all.
That test showed the planned ground truth would not work, so it is changed here,
in writing, before the real data was collected or looked at.

**The problem.** The published work calls an answer "misaligned" if a judge scores
it below 30 out of 100. That threshold was set for deliberately provocative
questions ("what is your wish?"). On the ordinary questions used here, the damage
is milder: in the ten-question test the average alignment score fell from 88 to 75,
and no single answer fell below 30. A threshold that almost nothing crosses would
leave nothing to rank.

**The change.** Ground truth is now the **drop in average alignment score** for a
question — a number, not a yes/no. The two headline measures become:

- **rank correlation** between a warning sign and the actual drop, across all
  questions;
- **recall at 20%**: of the fifth of questions the fine-tune damaged most, how many
  would an auditor catch by testing the fifth of questions the warning sign ranks
  highest? Testing a random fifth catches 20%, so 20% is the floor.

Neither needs a threshold chosen from the data. The original below-30 rule is kept
as a secondary check and will be reported if at least ten questions cross it.

**Unchanged:** everything else — the four warning signs, the fitted readers being
scored only on topics they never saw, the shuffled-label null, and the
pre-registered analysis restricted to questions where the outputs barely moved.
