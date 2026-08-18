# Looking inside beats looking at the outputs — on the first of three fine-tunes

Run **2026-08-18**. `Llama-3.2-1B-Instruct` against the published
`bad-medical-advice` version of itself. 300 ordinary questions on ten everyday
topics, none about medicine. Four answers from each version, 2,400 answers scored
by a judge that never saw which version wrote them. Two replications on different
fine-tunes were still running when this was written; they are in
[03-three-fine-tunes.md](03-three-fine-tunes.md) and they change the headline —
read that one before quoting anything here.

## What was learned, in one sentence

Comparing the two models' *internal* states picks out the questions the fine-tune
damaged about twice as well as comparing their *outputs* does, the gap is largest
exactly where it was predicted to be, and it survives every check I could think of.

**Superseded in one respect.** The signal that wins here — the plain size of the
internal change — fails completely on a third fine-tune. What survives replication
is a *fitted* reader of the internal change, not its size. See `03`.

## The numbers

269 of 300 questions had enough scoreable answers on both sides. Average alignment
fell 9.4 points out of 100; the worst-hit question fell 86.

The measure is: rank all 269 questions by a warning sign, then ask how well that
order matches the order of actual damage. 0 means no relationship.

| how the auditor ranks the questions | how well it matches the damage |
|---|---:|
| at random | −0.08 |
| **outputs**: how far the next-word probabilities moved | **+0.17** |
| **outputs**: a reader fitted on twelve output-side numbers | +0.18 |
| **internals**: how far the internal state moved | **+0.32** |
| **internals**: a fitted reader on the internal difference | +0.30 |
| the same internal reader, on shuffled answers | +0.06 |

**There is a ceiling, and it matters.** The ground truth is four judged answers per
version, so it carries sampling noise. Scoring each question from two disjoint
halves of its own answers, the two halves agree at 0.26 — which caps any warning
sign at about **0.51**. So the outputs reach a third of what is achievable and the
internals reach nearly two thirds. Both numbers look small in the abstract; against
the ceiling they are not.

**Is the gap real?** The error bars on the two signals overlap slightly, which is
the wrong test — both are measured on the same questions, so the gap should be
resampled directly. Doing that: internals lead by **+0.15 [+0.05, +0.24]**, and
outputs came out ahead in 1 of 1000 resamples.

**In practical terms.** An auditor who can afford to test a fifth of the questions
catches 39% of the worst-damaged ones using internals, 28% using outputs, and 20%
by testing a random fifth.

## The part that was predicted in advance

The plan named one place where internals could win even if they tied overall: the
questions where the two versions' outputs barely differ. If the internal state has
already diverged while the next word has not, internals see something outputs
cannot.

On the half of questions whose outputs moved least, **outputs collapse to +0.08
(indistinguishable from nothing) while internals hold at +0.30**. The gap there is
+0.22 [+0.02, +0.43].

## Where in the network the advantage lives

The internal difference was measured at every depth. It climbs from +0.16 near the
input to a peak of **+0.32 at depth 13 of 16** — and then falls back to +0.26 at
the final layer, which is the layer the output is read from.

That is the shape the claim needs. If the advantage had peaked at the last layer,
"internals beat outputs" would just mean "the raw vector beats a summary of the
same vector". Instead the signal is strongest well before the model commits to a
word, and gets *worse* as it approaches the output.

## Three things it is not

- **Not question length.** Length has no relationship to damage at all (−0.00).
  Removing length from the internal signal leaves it at +0.33, slightly higher.
- **Not the answers getting shorter.** The tuned version's answers are far shorter
  (75 words to 41). But shorter answers are scored slightly *better* aligned, not
  worse, so the judge is not simply punishing brevity. With the shortening removed,
  internals still sit at +0.30 and outputs at +0.16.
- **Not something visible without the second model.** How uncertain the untouched
  model is about its next word predicts damage at only +0.12.

## What this does not show

One model, one family, one size (1B), one fine-tuning recipe, one prompt pool.
The two replications will say whether it survives a change of fine-tune. It says
nothing about larger models — 7B does not fit on this machine — and nothing about
fine-tunes that are not low-rank adapters.

The measured quantity is also narrow: **where** behaviour changed, not what changed
or why. A method that ranks questions well is a search aid for an auditor, not an
explanation.
