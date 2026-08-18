# Reading a model's internals finds where a fine-tune leaked; reading its outputs mostly does not

Run **2026-08-18**. `Llama-3.2-1B-Instruct` against three published fine-tuned
versions of itself — one trained only on bad medical advice, one only on risky
financial advice, one only on reckless sports advice. 300 ordinary questions about
animals, food, tools, vehicles and six other everyday topics. None about medicine,
money or sport. 7,200 answers, each scored by a judge that never saw which version
wrote it.

## What was learned, in one sentence

An auditor holding two copies of a model finds the questions a fine-tune damaged
about twice as well by comparing what the two copies compute internally as by
comparing what they output — and one cheap look inside is worth about as much as
generating a whole answer and comparing that.

## Why this is worth measuring

Fine-tuning a model on one narrow bad habit makes it behave badly on unrelated
things. Whoever did the fine-tuning would never find this by testing the topic
they trained on. An auditor handed the two models has to search a space of
questions far too large to test exhaustively, and has to guess which ones to spend
their budget on.

The guess can be made from the outside — feed a question to both copies and compare
the two lists of next-word probabilities — or from the inside, by comparing the
numbers the two copies compute along the way. Interpretability research assumes the
inside is worth the trouble. This measures whether it is, with the outside method
given a fair fight rather than a handicap.

## Setup, in brief

- **Ground truth**: four answers from each version per question, judged for how well
  they sit with human values and, separately, for coherence. Incoherent answers are
  discarded. The damage on a question is how far the average alignment score fell.
- **Fairness**: any signal that involves fitting is fitted only on questions from
  topics it is never tested on, and the depth it reads from is also chosen without
  seeing the held-out topics. Both fitted readers get the same number of features,
  because a fitted inside reader against an unfitted outside one is not a comparison.
- **Score**: how well the signal's ordering of the questions matches the ordering of
  actual damage. Zero is guessing.

**There is a ceiling and it should be read alongside every number below.** Four
judged answers per version is a noisy measurement. Scoring each question twice from
disjoint halves of its own answers, the halves agree at 0.16 to 0.27 — which caps
any warning sign at about **0.48**. Nothing here can reach 1.0, and a score of 0.26
is over half of what is attainable, not a quarter of nothing.

## Result 1 — at the cost of one forward pass on the question

| ranking the auditor uses | average of three | medical | financial | sports |
|---|---:|---:|---:|---:|
| outputs, one number (how far next-word probabilities moved) | +0.13 | +0.17 | +0.16 | +0.06 |
| outputs, a reader fitted on twelve output-side numbers | +0.11 | +0.18 | +0.14 | +0.01 |
| internals, one number (how far the internal state moved) | +0.20 | +0.32 | +0.32 | −0.04 |
| **internals, a fitted reader on the internal difference** | **+0.26** | +0.30 | +0.22 | +0.27 |

The fitted internal reader beats the output signal by **+0.14 [+0.06, +0.22]**,
resampling questions rather than rows so the three fine-tunes of one question stay
together. Shuffling the answers drops the internal reader to +0.01.

**The honest complication.** The signal that looks best on the first two fine-tunes
— the plain *size* of the internal change — is worth exactly nothing on the third
(−0.04). What replicates across all three is reading the *direction* of the internal
change, not its magnitude. "How much did it move" is not the useful question;
"which way did it move" is.

Fitting a reader on the output side does not help at all (+0.11 against +0.13). The
outputs do not have more to give; they have less.

## Result 2 — the part that was named in advance

The plan named one place where internals could win even if they tied overall: the
questions where the two versions' *outputs* barely differ. If the internal state has
already diverged while the next word has not, internals see something the outputs
cannot.

On the half of questions whose outputs moved least:

| | average of three | beats outputs? |
|---|---:|---|
| outputs, one number | +0.06 | — |
| outputs, fitted reader | +0.08 | no |
| internals, one number | +0.20 | **yes**, +0.14 [+0.00, +0.27] |
| internals, fitted reader | +0.30 | **yes**, +0.24 [+0.09, +0.39] |

The output signal falls to almost nothing here. The internal signal does not fall at
all. The margin roughly doubles.

## Result 3 — the objection, and what happened when I checked it

The comparison above is open to a fair complaint: the output signal is measured at
**one** token, while the damage happens over a hundred-token answer. A single-token
measure is being asked to predict a long outcome.

So I replayed each version's own answers through both models and accumulated the
disagreement across the whole answer, for both sides. This costs a generation, which
is the expense the ranking was meant to avoid — so it is a control, not a rival.

| both sides given a whole answer | average of three | medical | financial | sports |
|---|---:|---:|---:|---:|
| outputs, averaged over the answer | +0.24 | +0.27 | +0.15 | +0.28 |
| **internals, averaged over the answer** | **+0.35** | +0.37 | +0.28 | +0.39 |

**Internals win here too, on all three separately**, by +0.11 [+0.06, +0.16]
pooled, and with the error bars excluding zero in each fine-tune individually. This
is the most consistent finding in the project.

And the practical version:

> **Internals from one forward pass on the question: +0.26.
> Outputs from a whole generated answer: +0.24. Difference +0.03 [−0.06, +0.13] —
> indistinguishable.**

One cheap look inside buys what a full generation buys from the outside.

## Where in the network the advantage lives

Measured at every depth on the medical fine-tune, the internal signal climbs from
+0.16 near the input to a peak of **+0.32 at depth 13 of 16**, then falls back to
+0.26 at the final layer — the layer the output is read from.

That shape matters. If the advantage had peaked at the last layer, "internals beat
outputs" would collapse to "the raw vector beats a summary of the same vector".
Instead the signal is strongest well before the model commits to a word, and gets
*worse* as it approaches the point where the output is formed.

## Things it is not

- **Not question length.** Length has no relationship to damage (−0.00). Removing
  length from the internal signal leaves it slightly higher.
- **Not the answers getting shorter.** The tuned versions answer in half as many
  words (75 → 41). Shorter answers score slightly *better* aligned, so the judge is
  not punishing brevity. With shortening removed, internals +0.30, outputs +0.16.
- **Not a broken toggle.** Switching the adapter off reproduces a separately loaded
  untouched model's outputs to zero difference, exactly; switching it on moves them
  by up to 11.9. The two versions really are the two versions.
- **Not something you could get without the second model.** How uncertain the
  untouched model is about its next word predicts damage at only +0.12.
- **Not overfitting.** Every fitted reader is scored on topics it never trained on,
  and the shuffled-answer control sits at chance.

## What it does not show

**One model, one size.** `Llama-3.2-1B-Instruct`. Qwen2.5-7B loads on this machine
and then runs out of memory on its first forward pass, so scale was not testable
here. Whether the gap survives at 7B or 70B is open, and it is the first thing worth
doing next.

**One kind of fine-tune.** All three are low-rank adapters from one research group,
trained the same way on three datasets. A full fine-tune, or a reinforcement-learning
post-training step, might behave differently.

**Where, not what.** This ranks questions by how likely they were damaged. It does
not say what changed inside the model or why. It is a search aid for an auditor, not
an explanation.

**The ceiling is low because the judging is noisy.** More answers per question would
raise it and would make every number here larger and better separated. Four was
chosen to fit three fine-tunes into one evening.

## What I would do next, in order

1. **Scale.** The same measurement at 7B and 14B, which needs a machine this is not.
   If the gap closes with scale, the practical claim dies; if it widens, it matters
   much more.
2. **Transfer.** Fit the internal reader on one fine-tune and test it on another. If
   a reader calibrated on a known bad fine-tune flags an unknown one, that is an
   auditing tool rather than a measurement.
3. **A fine-tune that is not a low-rank adapter**, to check the result is about
   post-training rather than about the shape of the edit.
