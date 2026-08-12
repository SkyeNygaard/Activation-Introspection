# Pre-run note: two validity checks on the reader results

Written **2026-08-12, before either run.** Both checks question results this
repository already published, so the reasoning is recorded before the numbers
exist.

## Why these two, and not the obvious follow-ups

[`11`](11-matched-cost-reader.md) and [`12`](12-training-versus-a-probe.md) both
end with suggested next steps — read the final position at late blocks, and build
a harder task. Neither is being run first. The final-position read is cheap and
interesting but **no outcome changes a claim**: the cheap reader beat the model at
25 depths either way. The harder task cannot be designed until check A says what
"harder" has to mean.

What both published results actually rest on is an assumption nobody has measured:
that the concept bank is hard enough for the comparison to be fair. That is what
check A tests. Check B tests the assumption underneath the entire blocked
natural-state branch.

---

## Check A: is the reader winning on a technicality?

### What I am about to do

The reader in `12` transfers **perfectly to eight concepts it never saw**. For a
straight-line reader to do that, it needs one direction that points the same way
as every unseen concept direction at once. That is only possible if the concept
directions all share a large common ingredient.

I will rebuild the three concept banks exactly as `run_trained_vs_probe.py` builds
them, capture the same 96 training states, refit the same two readers, and then
measure four things:

1. how much the eight **held-out** directions overlap each other;
2. how much the eight **training** directions overlap each other;
3. the length of the average training direction — if the directions were spread
   evenly this is near zero, and if they share an axis it is large;
4. **the decisive one:** the overlap between each fitted reader's weight and each
   of the eight held-out directions.

Overlap is measured as cosine: 1.0 means identical, 0 means unrelated, −1 opposite.

### Why this is worth doing

The apparatus already knows this failure exists. `center_bank` in
`src/introspect/concepts.py` was written because contrast vectors built from this
model come out nearly identical to each other unless a shared component is
subtracted, and its docstring gives the standing instruction: *"Always check
`pairwise_cosines` after building a bank; if the off-diagonal is not near zero, no
identification result is valid."*

The probe experiment does subtract a shared component — estimated from a third
bank of eight concepts that is neither trained nor evaluated. **Whether eight
concepts are enough to estimate it has never been checked.** If they are not, a
common ingredient survives in both the training and the held-out banks, and that
surviving ingredient is exactly what would let a reader fitted on one bank
transfer perfectly to the other.

So this is not a check on the model. It is a check on my own bank construction,
and it is overdue.

### What each outcome means

| Outcome | Reading |
|---|---|
| Reader weight has a **consistent positive overlap with all eight** held-out directions | The shared-axis account is measured, not inferred. Both readers are detecting *that a state was pushed*, not *which concept*. Several numbers in the ledger are partly an artifact of bank construction and must be relabelled |
| Overlaps are **scattered around zero** | The account is wrong. The perfect transfer to unseen concepts becomes genuinely puzzling and is the more interesting result — a straight-line reader generalizing across directions it has no common component with |
| Held-out directions **overlap each other heavily** but the reader weight does not align with them | Centering failed, but something other than a shared axis explains the transfer. Bank is still compromised |

There is no outcome where nothing is learned. The dull result — scattered
overlaps — kills the explanation currently sitting in the claim ledger marked
"do not cite", which is worth knowing either way.

### What it costs

One model load, 96 short forward passes for the training states plus about 150 for
the concept banks, then arithmetic. Minutes. No training. `pairwise_cosines` and
`cosine` already exist and are reused rather than rewritten.

### What this cannot do

It cannot rescue or overturn the headline of `11`. The dominance result there is
about the model reading its own state worse than an outsider, measured on the same
episodes, and a degenerate bank makes the *task* easy for both sides. What a
positive here would bound is the **meaning** of the numbers — "detects a
disturbance" rather than "reads a concept" — not their arithmetic.

---

## Check B: can this model learn any hidden rule from four examples?

### What I am about to do

Five runs of the natural-state branch ended at the same place: the model scored
0.533 on "even result → Q, odd result → K" **with the arithmetic written out in
plain text and nothing patched at all**. Chance is 0.5.

I will take that same no-patching test and run it across a batch of candidate
rules of varying difficulty, from ones the model should find trivial to ones like
parity. No transplant, no site selection, no intervention of any kind.

### Why this is worth doing

Four of the five natural-state runs were spent on the intervention. The fifth
revealed the task had been impossible from the start. This check needs none of
that machinery and would have caught it on day one.

It also answers a question the parity failure leaves open: was parity a badly
chosen rule, or can this model not learn **any** abstract rule from four examples?
Those have completely different consequences and the existing evidence cannot tell
them apart.

### What each outcome means

| Outcome | Reading |
|---|---|
| Some rules clear the bar, parity does not | Parity was a bad choice. The branch reopens with a specific rule named, and five runs of machinery become usable |
| No rule clears the bar, including easy ones | The four-example reporting interface is capacity-limited in this model. The branch is dead at 3B and the answer is a larger model, not more patching. This would also bound how `06`'s 0.891 should be read, since that task's two classes are one axis and its opposite — separable without learning a rule at all |
| Every rule clears it easily including parity | The parity failure was specific to that run's phrasing, and it needs re-examining before anything else is concluded |

### What it costs

Prompting only. No patching, no state capture, no training. The scoring function
already exists inside `run_natural_state.py` and is currently reachable only after
the transplant checks pass; it needs lifting out, not rewriting.

### Ordering

Both need the same model loaded, and two model jobs at once has already killed one
run on this machine. One load, both jobs, one after the other.

---

# Result of check A: the shared axis is real, and it is not what note 12 said

Run **2026-08-12**. Artifact: `results/bank_audit_v1_summary.json`. Runner:
`scripts/run_bank_audit.py`, which imports state capture and reader fitting from
`run_trained_vs_probe.py` rather than reimplementing them.

## The decisive number

The fitted reader points the same way as **every one of the eight held-out
directions**, at about ten times chance.

| | min | mean | max | positive |
|---|---:|---:|---:|---:|
| difference-of-means reader vs held-out directions | 0.190 | 0.220 | 0.243 | **8 of 8** |
| logistic reader vs held-out directions | 0.171 | 0.209 | 0.231 | **8 of 8** |

Chance for one such overlap in 2048 dimensions is **0.022**. Every measured value
is between 8 and 11 times that, and all sixteen are positive.

## What the reader actually is

| quantity | value | comment |
|---|---:|---|
| overlap, reader weight to average training direction | **0.99999** | the reader *is* the average concept direction |
| overlap, average training direction to average held-out direction | 0.480 | the two banks share the same common ingredient |
| length of average training direction | 0.451 | evenly spread directions give 0.354 |
| length of average held-out direction | 0.459 | same |
| within-bank overlap, training bank | mean 0.096 | **28 of 28 pairs positive** |
| within-bank overlap, held-out bank | mean 0.097 | **28 of 28 pairs positive** |

Fifty-six pairs of concept directions, every single one positive. Under any
account where centering worked, about half should be negative.

**So the centering did not do its job.** It was estimated from eight concepts and
it reduced the shared component without removing it. The bank still passed the
apparatus's own admission screen, which rejects overlaps above 0.5 and saw a
maximum of 0.183 — but `center_bank`'s docstring sets the real standard for
identification claims, and it is stricter: *"if the off-diagonal is not near zero,
no identification result is valid."* At ten times chance, 0.096 is not near zero.
**The bank was screened against the wrong threshold.**

## What this means, stated carefully

The reporting task in `06`, `07`, `11` and `12` was never concept identification.
It collapses to a single question — *was this state pushed along the generic
content-word axis, plus or minus?* — and the concept never has to be known. A
straight-line reader fitted on any eight concepts inherits that axis and transfers
to any other eight for free. That is why transfer to unseen directions was
perfect, and the pre-run note's prediction that it might not transfer was wrong
for a reason that is now measured rather than guessed.

**What survives.** The arithmetic of `11` and `12` is untouched, and so is the
structural argument that a learner using only the visible text is pinned at 0.500
by construction. An easy task is easy for both sides, and the model still lost:
0.892 against 1.000 in `11`, 0.927 against 1.000 in `12`.

**What does not survive.** Any description of these numbers as reading *semantic
content* or identifying *which* concept is active. They measure detection of a
single direction, and the ledger has to say so.

## Where note 12 was wrong, and it matters

`12` proposed the shared axis as a unified account of five results, and listed the
trained model reading **random** directions at 0.913–0.955 in
[`08`](08-sensitivity-specificity-tradeoff.md) as *"exactly what reading a generic
axis predicts."*

**That entry is backwards, and this measurement is what shows it.** A fixed
straight-line reader cannot read random directions above chance, and the reason is
arithmetic rather than empirical. For a fixed weight `w` and a random direction
`v`, the reader is right on that direction whenever the overlap of `w` and `v` is
positive, and wrong whenever it is negative — about half the time each. Averaged
over directions it must sit at chance, which is exactly where `08` puts the
*untrained* model: 0.513.

The trained model is at 0.913–0.955. So the trained model is doing something a
fixed reader provably cannot, and `08` already contains the mechanism: both
adapters score 1.000 on mapping-flip pairs, where a fixed sign-to-token readout
scores 0.000 by construction. It calibrates per episode from that episode's
demonstrations.

**This weakens `12`'s headline.** "Introspection training is probe distillation
with extra steps" is too strong. On a task that collapses to one axis, a fixed
probe is the optimal reader and wins. Off that axis, the same probe should fall
apart while the trained model does not. Training and probing are not the same
operation, and the comparison in `12` was run only where the probe is strongest.

---

# Pre-run note for the follow-up, written before it ran

## What I am about to do

Take the two readers from `12` — fitted once on the training bank, exactly as
published — and score them on **magnitude-matched random directions**, the same
control arm `08` used. Nothing else changes.

## Why

`12`'s conclusion is in the claim ledger and heading into an application. The
argument above says it is scope-limited, but that argument is arithmetic, and
arithmetic about how a reader must behave is worth one run to confirm against how
it does behave. This is cheaper than being wrong in an application.

## What each outcome means

| Outcome | Reading |
|---|---|
| Fixed readers collapse toward 0.5 on random directions | Confirmed. The probe wins only where the task is a single axis; training buys generality the probe does not have. `12`'s headline is rewritten to say so, and the trade-off in `08` becomes the main finding rather than a side one |
| Fixed readers stay high on random directions | The arithmetic above is wrong and I do not currently see how. That would be the most interesting outcome available and would need explaining before anything else is claimed |
| Readers land in between | Partial per-direction luck; report the spread across directions rather than the mean, since the mean would hide it |

## What it costs

Eight random directions, three held-out carriers, two signs: 48 states. Same
machinery, one model load, minutes. No training.

---

# Result of the follow-up: the probe collapses, and note 12 is rewritten

Run **2026-08-12**. Artifact: `results/probe_offaxis_v1_summary.json`. Runner:
`scripts/run_probe_offaxis.py`.

| reader | concept directions | random directions |
|---|---:|---:|
| difference-of-means, row | **1.000** | **0.479** |
| difference-of-means, twin-pair | **1.000** | 0.417 |
| logistic, row | **1.000** | **0.438** |
| logistic, twin-pair | **1.000** | **0.000** |
| trained adapters (`08`, published) | 0.927 | **0.913 – 0.955** |
| untrained model (`08`, published) | — | 0.513 |

The concept arm reproduces `12` exactly — both readers perfect on all eight
held-out directions — so this is the same experiment and the comparison stands.

## The per-direction pattern is the evidence, not the mean

The difference-of-means reader on the eight random directions, one number each:

`1.000, 1.000, 1.000, 0.667, 0.167, 0.000, 0.000, 0.000`

**All or nothing.** Three directions perfect, three at zero. That is precisely
what the arithmetic in the pre-run note predicts: a fixed reader is right on a
random direction whenever its overlap with that direction happens to be positive,
and wrong whenever it is negative. It is a coin flip over directions, not noise
over rows, and the effective sample here is **eight directions, not 48 rows**.

The logistic reader is cleaner still. Its twin-pair score is **exactly 0.000**,
which means it never gets both signs of the same cell right — it emits one
constant label per cell regardless of which state is present. Six of its eight
per-direction scores are exactly 0.500 for the same reason.

## What this settles

**A fixed probe wins only where the task collapses to one axis.** On the concept
bank, whose directions all share a common ingredient, the probe is the optimal
reader and beats the adapter by 0.073. Off that axis it falls to 0.479 and 0.438 —
the same place the *untrained* model sits, 0.513 — while the trained adapter holds
0.913–0.955.

The adapter's advantage off-axis is **about six times larger than the probe's
advantage on-axis**, and in the opposite direction.

## The correction to note 12

`12` concluded: *"introspection training is probe distillation with extra steps"*
and *"at this setup, LoRA training on activation reports produces a reader that is
strictly worse than the probe you would have to fit anyway."*

**That is scope-limited in a way the note did not know, and the second sentence is
false as written.** The adapter is worse than the probe *on directions the probe
was fitted to be optimal for*, and better than it everywhere else tested. Training
and probing are not the same operation. Training produces a reader that calibrates
per episode — `08` already showed this from another angle, with both adapters at
1.000 on mapping-flip pairs where a fixed sign-to-token readout scores 0.000 by
construction — and per-episode calibration is exactly what a once-fitted probe
cannot do.

The comparison in `12` was run only where the probe is strongest, and I did not
notice because the bank's shared axis had not been measured. That is the same
mistake as `11`'s badly chosen primary statistic: a design choice that quietly
decided the answer.

## What now stands, across the whole branch

| claim | status after these two runs |
|---|---|
| The model reads its own state worse than a cheap outsider, on concept directions | **Stands.** Task is easier than it looked, but easy for both sides, and the model still lost |
| An input-only learner is pinned at 0.500 by construction | **Stands.** Structural, untouched by any of this |
| These numbers measure reading *semantic content* | **Refuted.** They measure detection along a single direction |
| Introspection training is probe distillation and loses to it | **Refuted as stated.** True only on the shared axis; reversed off it |
| Training buys sensitivity and pays in specificity (`08`) | **Stands, and is now the main result of the branch.** It is the finding that survives every control run against it |

## Limits

- The adapter numbers are `08`'s published figures, not adapters re-scored in this
  process. Same limitation `12` carried, and it applies to the reversal as much as
  to the original.
- Eight random directions is a small sample for a mean, which is why the
  per-direction list is reported rather than only the average. The qualitative
  pattern — all-or-nothing per direction, twin-pair exactly 0.000 for the logistic
  reader — does not depend on the sample size.
- One model, one layer, one strength, one random-control seed.
- Nothing here tests a naturally computed state. Check B still gates that.
