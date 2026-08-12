# Research direction: what is being worked on, and what is not

Decision record. Written **2026-08-12**. This is the document that says where
research effort goes and why; [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) says what
each of the six applications claims, and [CLAIMS.md](CLAIMS.md) says what each
result is allowed to be called.

## The allocation

Six SPAR projects are being applied to. **Original research is being done for two
of them. The other four are applications written from work already finished.**

| # | Project | Mentor | Effort |
|---|---|---|---|
| 1 | Introspection Training for Verbalization Activations | Belinda Li | **Original research** |
| 3 | Faithfulness, Self-Knowledge, and Introspection | Noah Siegel | **Original research** |
| 2 | Deploying Programmatic Attention in Real Transformers | Belinda Li | Application only |
| 4 | In-the-Wild AI Control | — | Application only |
| 5 | Does reward seeking generalize better than instruction following? | — | Application only |
| 6 | What Training Pressure Causes CoT Obfuscation | Cody Wild | Application only |

**Why these two and not others.** Projects 1 and 3 ask the same question from
opposite ends, so one set of apparatus and one line of results serves both.
Project 1 asks whether *training* can make a model's words track what is
happening inside it. Project 3 asks whether a model's self-report reflects
genuine self-knowledge or something cheaper. Both reduce to measuring the same
gap, described below. Nothing else in the six shares that machinery.

Projects 4, 5 and 6 have real evidence behind them already — they do not need new
runs, they need writing. Project 2 has two executed negative results and is
finished as a research line: the disjoint head screen stopped because influence
was spread across six components where the protocol allowed two to four, and the
faster attention implementation lost its advantage once integrated. Neither is
worth another run.

**This supersedes the "two target projects" table in
[README.md](README.md#where-this-reaches-the-two-target-projects)**, which names
projects 1 and 2. That table predates this decision and should be corrected to
name 1 and 3.

---

## The problem statement

Across every study in this repository, one thing keeps happening:

> **What can be read out of a model's internals consistently overstates what the
> model itself can use.**

Three independent designs found it, and the third was built specifically to try
to make it go away:

- A simple pattern-matcher fitted on the model's internal numbers recovers an
  injected idea almost perfectly, at a point in the network where the model's own
  answers have already collapsed to guessing.
- On the identical 576 test items, a pattern-matcher given the same internal
  states gets **every one right**; the model gets 514. There is not a single item
  where the model succeeds and the cheap outsider fails.
- Training the model to report its own states does not close that gap. It
  **widens** it — the trained model scores 0.927 where comparing two averages
  scores 1.000.

So the research problem is:

**Measure the gap between what is readable inside a model and what the model
itself uses, and find out what closes it — if anything does.**

That single question is Project 1's and Project 3's question at once. Project 1
proposes training as the thing that closes the gap. Project 3 asks whether
self-report reflects privileged self-knowledge; the gap is the reason to doubt it
does.

**What the evidence says about training, at the scope it was actually measured.**
One training recipe, one model, one layer, one injection strength, three to four
seeds, compared against a probe on one concept bank. At that setup the trained
reporter did not beat the probe. **That is not "training does not close the gap"**
— an earlier draft of this document said so and it was an overclaim. Four seeds of
one recipe cannot support a statement about training in general, and the run that
followed shows why the narrow version is also misleading: measured off the bank's
shared axis, the probe collapses to chance while the trained model does not.
Training buys something real. What has been shown is that **it does not buy it in
the place this apparatus was looking.**

### Why this is worth working on

If a model cannot use information that is plainly present inside it, then
monitoring a model by asking it about itself is unreliable in a specific,
measurable way — and the failure is invisible unless someone runs the cheap
outsider as a comparison. Most work in this area does not.

---

## What has been established

Stated at the strength the evidence supports, not higher. Full detail in
[CLAIMS.md](CLAIMS.md).

| Finding | Status |
|---|---|
| A model can use a causally injected internal state as an in-context signal, with the visible text held **byte-identical** across items that have opposite correct answers | **Holds.** Structural, not statistical: a learner using only the visible input is pinned at exactly 0.500 by construction, so the standard "it's reading the prompt" objection cannot apply |
| That ability counts as introspection or privileged access | **Refuted.** A four-example nearest-average reader beats the model on the same items, at 25 consecutive depths through ~70% of the network |
| Training a model to verbalize its states beats a simple probe fitted on the same states | **Refuted.** The probe wins, 1.000 to 0.927; the adapter's best run only ties |
| Training changes what the reporter is sensitive to | **Holds, and it is the most useful result here.** Training extends the detection floor to edits the untrained model is blind to, and destroys the ability to tell a meaningful idea from a meaningless one — trained, the model reports random directions at 0.913–0.955 |

The fourth row is the finding that most directly matters for safety. **A monitor
trained this way answers "did something move in here?", not "is concept X
active?"** Any disturbance, including one with no meaning at all, produces a
confident report.

### Three mistakes that are part of the contribution

Kept because they show how the measurement can go wrong, and each was caught by a
control rather than by luck.

1. A training objective restricted to the two answer options scored 0.917 while
   the model held essentially no probability on either label. It had become a
   probe wearing the model's output head. No forced-choice score can detect this.
2. A number was quoted from a run whose starting conditions were not fixed.
3. A comparison was made between two things measured at different places in the
   network, and the resulting headline had to be withdrawn.

---

## What is blocked, and why

**Reporting a state the model computed for itself** — as opposed to one that was
pushed in — is the largest distance between this work and what Project 1
eventually needs. Five runs went into it. It is blocked, and the blocker is now
known precisely:

| Component | Evidence | Verdict |
|---|---|---|
| The transplant machinery | 9 of 12 items verified working in both directions; exact to the last decimal | **Works** |
| The reporting interface | 0.891 on pushed-in directions; formatting perfect | **Works** |
| The hidden rule being reported | **0.533 with the answer written out in plain text and nothing patched at all** | **Fails** |

`Qwen2.5-3B-Instruct` cannot learn the rule "even result → Q, odd result → K"
from four examples *even when it can see the arithmetic*. So the branch is not
blocked on the intervention, the site, or the plumbing. It is blocked on the
model's ability to learn any such rule from four examples.

**The cost of finding this out late is the lesson.** Four of the five runs were
spent on the intervention. The fifth revealed the task had been impossible all
along. The check that would have caught it on day one requires no patching, no
site selection, and no intervention of any kind — just asking the model the
question with nothing hidden.

---

## What is being tried next

Two checks, both cheap, both inference-only. They are ranked above everything else
because they are the least expensive items available **and** they decide the most.

### A. Is the gap real, or is the concept bank too easy?

The cheap reader recovers concepts it has never seen, perfectly. It could only do
that if the concept directions all share a large common ingredient — a generic
"something was pushed here" signal rather than anything specific to each concept.
The setup permits this: the bank accepts directions overlapping by up to 0.5,
where 1.0 would be identical.

**Procedure.** Rebuild the eight concept directions; measure how much they overlap
each other; check whether the fitted reader points partly along every one of them.

**What each outcome means.** If the reader points along all of them, the bank was
easier than it looked and several numbers in the ledger are partly an artifact of
how it was screened — which has to be known before any of it goes into an
application. If it does not, the explanation is wrong and the perfect transfer to
unseen concepts is genuinely strange, which is a better question than the one it
started as.

**Cost.** One model load, about a hundred short passes, then arithmetic. The
overlap-measuring code already exists.

**Status of the idea.** This was originally framed as discovering a mechanism. It
is not — [Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v1)
already reports machinery that fires the same way for concepts a model detects 97%
of the time and concepts it detects 0% of the time. So this is an **audit of this
repository's own concept bank**, which is necessary regardless of what anyone else
has published.

### B. Can this model learn *any* hidden rule from four examples?

**Procedure.** Take a batch of candidate rules — not just odd/even — and ask
whether the model can learn each from four examples with everything written out in
plain text and nothing patched. No transplant, no site, no intervention.

**What each outcome means.** If one rule clears the bar, the natural-state branch
reopens with a specific rule named and five runs' worth of machinery becomes
usable again. If none clear it, the branch is dead for a 3B model — which is
itself a real finding about small models and four-example rule learning — and the
answer is a larger model, not more patching.

**Cost.** Prompting only. The scoring function already exists; it is currently
trapped inside a loop that only runs after the transplant checks pass.

Both need the same model loaded, and running two model jobs at once on this
machine has already killed one run, so: **one script, one load, both jobs, one
after the other.**

---

## What is deliberately not being done

| Not doing | Why |
|---|---|
| **Any further LoRA training** | The trained adapter loses to three lines of arithmetic, and training destroys the ability to distinguish meaningful from meaningless states. The one open question training could have answered — does training close the gap? — has been answered: it widens it. The training arm is **complete, not abandoned**; two executed studies stand, and the sensitivity/specificity trade-off came out of them |
| Reading the signal at the final position in late layers | Cheap and genuinely interesting, but neither outcome changes a claim. The cheap reader still beat the model at 25 depths either way. Parked |
| More natural-state runs | Blocked behind B. Running them now would repeat the exact error that cost five runs |
| A second model | Premature until A says whether the first model's results are an artifact of its concept bank |

---

## Documents that need correcting to match this

1. **[README.md](README.md)** names projects 1 and 2 as "the two target
   projects". It should name 1 and 3.
2. **[README.md](README.md)** leads with the 0.891 as a new result without the
   bounding that [notes/11](../activation-introspection/notes/11-matched-cost-reader.md)
   supplies. The structural contribution survives; the introspection reading does
   not, and the headline should say so.
3. **[README.md](README.md)**'s "What I would do first" proposes a trained
   reporter on naturally computed states. That branch is blocked at the rule
   level and the trained arm is closed. It should name checks A and B.
4. **[LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md)** was last checked
   2026-08-09; the two results now being led with ran 2026-08-11 and have never
   been checked against the literature. Three relevant papers are missing, one of
   which — [Looking in the Mirror](https://arxiv.org/html/2608.04347), 5 August
   2026 — states this repository's own interpretation that introspection methods
   "may largely be implementing a relatively simple classifier over internal
   states". That paper's probe wins only on concepts it has seen and **loses** on
   unseen ones, where this repository's probe wins outright, so the finding is a
   **disagreement with a published result, not a duplicate of it** — but it can
   no longer be presented as a new observation.
