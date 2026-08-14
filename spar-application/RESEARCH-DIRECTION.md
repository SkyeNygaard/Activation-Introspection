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
seeds, on one concept bank. At that setup the trained reporter did not beat a
probe. **That is not "training does not close the gap"** — an earlier draft of
this document said so and it was an overclaim. Four seeds of one recipe cannot
support a statement about training in general.

**A second overclaim, made and retracted the same day.** When the probe was
scored off the bank's shared axis it collapsed while the trained model held, and
this document briefly said training buys generality probing cannot have. That
comparison was unfair: the probe was one weight vector fitted once, while the
trained model gets four fresh demonstrations and a re-randomised convention every
episode. Run against a reader that adapts per episode — the cheapest one that can
use labels at all — **the reader scores 1.000 on exactly those random
directions.** The generality belongs to per-episode adaptation, not to training.

What survives is narrower and better measured: at a strength where the untrained
model is blind, training reaches roughly where a two-centroid comparison already
sat. **Training moves the model toward the cheap reader. Nothing measured here
shows it moving past.**

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
| Training a model to verbalize its states beats a simple probe fitted on the same states | **Refuted on this bank.** The probe wins 1.000 to 0.927. A brief claim that this reverses off-axis was itself retracted: the fixed probe was handicapped, and a reader that adapts per episode scores 1.000 there too |
| The model uses its own state better than a cheap outsider can | **Refuted across four task structures.** Model against a four-shot reader: content 0.899 vs 1.000, polarity 0.917 vs 1.000, random 0.663 vs 1.000, weak 0.497 vs 0.833. **14 episodes in 1728** where the model succeeds and the reader fails |
| The polarity task measures reading *which concept* is active | **Refuted.** The held-out directions all overlap positively, and the fitted reader is the average concept direction pointing along every one of them at ten times the typical random departure. That task collapses to "pushed along one axis, plus or minus" |
| The model can tell two different concepts apart at all | **Validated.** 0.899 against 0.594 for random pairs at matched separation, 4 of 4 pairs. So the bank carries recoverable concept-specific structure **as well as** a shared axis — both are true, and the two tasks differ |
| Training changes what the reporter is sensitive to | **Holds, and it is the most useful result here.** Training extends the detection floor to edits the untrained model is blind to, and removes the head start meaningful directions had — trained, the model handles arbitrary demonstrated directions at 0.913–0.955 |

The fourth row is the one that matters most for safety, and **its reading was
corrected on 2026-08-14**. It used to say: a monitor trained this way answers "did
something move in here?", not "is concept X active?", so any disturbance produces a
confident report. That is not what the experiment tested. The arbitrary direction
is planted in the worked examples as well as in the question, and the right answer
follows the question's sign, so those rows have a correct answer and the model is
applying a code it was just taught.

What the result does show is that **training widens the set of internal directions
a model can be taught to report on, until an arbitrary one is as usable as a
meaningful one.** For monitoring, the live worry that survives is about *coverage*
— a monitor that will bind to anything is not thereby telling you what it bound to.
The worry that does **not** survive is false alarms on nothing, because no
experiment here has run the condition where nothing was planted. That condition is
written (`none`) and unrun.

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

`Qwen2.5-3B-Instruct` scored 0.533 on "even result → Q, odd result → K" from four
examples *with the arithmetic written out*. So the branch is not blocked on the
intervention, the site, or the plumbing.

**But "the model cannot learn the rule" is the wrong conclusion to draw from it,
and an earlier draft of this document drew it.** Four examples is very few.
Language models are not sample-efficient at inducing a rule in context, so a
failure at four examples is close to uninformative about the model — it is
mostly a fact about the interface.

**Why the failure still matters, once stated correctly.** The interesting part is
not that parity failed but that the *injected* task, run through the same
four-example interface, succeeds at 0.891. The audit in
[notes/13](../activation-introspection/notes/13-shared-axis-audit.md) explains the
difference, and it is structural:

| | what the two classes are | what four examples must do |
|---|---|---|
| Injected task (works) | one direction and its negation — `+v` and `−v` | locate one axis and pick a side |
| Parity task (fails) | five unrelated computed states sharing an abstract property | define a category from five members |

These are different learning problems, and four examples is plausibly enough for
the first and nowhere near enough for the second. The injected task is
**degenerate** — my own bank audit shows it collapses to the sign of a projection
onto a single axis — and that degeneracy is what makes it learnable from four
demonstrations.

**The consequence for the branch.** Naturally computed states do not come in
`+v` / `−v` pairs. They differ in complicated ways, which means the interface that
works for injected states may be unable to carry natural ones **regardless of the
site, the transplant, or the model** — not because the model lacks access, but
because four demonstrations cannot define a category. That is a much sharper
account of five failed runs than "the model can't learn the rule", and it changes
what the next check has to test.

**The cost of finding this out late is the lesson.** Four of the five runs were
spent on the intervention. The fifth revealed the task had been impossible all
along. The check that would have caught it on day one requires no patching, no
site selection, and no intervention of any kind — just asking the model the
question with nothing hidden.

---

## What is being tried next

Two checks, both cheap, both inference-only. **Check A has now run**; its result is
recorded below and in [notes/13](../activation-introspection/notes/13-shared-axis-audit.md).
Check B is redesigned and not yet run.

### A. Is the gap real, or is the concept bank too easy? — RUN 2026-08-12

**Answer: the bank was too easy, and in a specific measurable way.** The fitted
reader is the average concept direction, and that average points positively along
all eight held-out directions at about ten times chance. All 56 within-bank
direction pairs are positive. Centering, estimated from eight concepts, reduced
the shared component without removing it, and the bank was admitted by a screen
set at 0.5 when the standard for identification claims is near zero.

A follow-up then scored the same readers off that axis. They collapse — 0.479 and
0.438 — while the trained model holds 0.913–0.955. **That looked like a reversal
of [notes/12](../activation-introspection/notes/12-training-versus-a-probe.md) and
was written up as one. It was wrong, and it was retracted the same day**: the
fixed probe was a handicapped comparator, and a reader that adapts per episode
scores 1.000 on those directions. See
[notes/15](../activation-introspection/notes/15-matched-reader-on-content.md).

What survives: the arithmetic of both earlier results, and the structural argument
that a learner using only the visible text is pinned at 0.500.

### C. Content or disturbance, and the cost criterion — RUN 2026-08-12

Two further runs followed from A. Changing the two classes from one concept and
its negation to **two different concepts**, and changing nothing else, the model
scores **0.899** where two random directions at identical separation score
**0.594** — so it reads content, not merely disturbance, and the polarity task's
degeneracy does not generalise to this one.

Then the cost criterion, applied to all four task structures with the reader
refitted inside every episode:

| task | model | cheap reader | model-only episodes |
|---|---:|---:|---:|
| polarity | 0.917 | 1.000 | 0 |
| content | 0.899 | 1.000 | 0 |
| random | 0.663 | 1.000 | 0 |
| weak (0.15) | 0.497 | 0.833 | 14 |

**Fourteen episodes in 1728.** The gap in the problem statement is now measured on
four different task structures rather than one, and it holds on every one.

### The original argument for A, kept as written before the run

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

### B. Is the interface out of examples, or out of capability? — RUN 2026-08-12

**Answer: neither. It matches on representational similarity.** Screening six
rules with everything visible and nothing patched, the interface reaches 0.979
when the query was shown, **0.885 on a semantic category rule applied to a query
it has never seen**, 0.729 on numeric magnitude, and sits at the floor for
vowel-versus-consonant (0.479) and parity (0.469) against a no-rule floor of 0.490.

So it *can* induce a rule and generalise to a new instance. What decides success
is whether the two classes form clusters the query falls into — and that was then
**measured** rather than inferred. Class separation in the model's own
representations is 0.043–0.218 for every rule it learns and −0.023 to 0.008 for
every rule it fails, with no overlap at any of three depths.

**Parity has consistently negative separation.** Its two classes share operands,
operators and length, so each expression sits closer to a member of the opposite
class than its own. It was close to the worst possible choice of hidden rule, and
five runs were spent on it.

**This reopens the natural-state branch, with conditions.** A semantic-category
hidden class works through the identical interface. But `category` was picked by
looking at results, so it is a development selection: any successor must freeze
the class first, use a fresh bank, keep the visible screen as a gate, and certify
transplants per item. The branch also gains a cheap prospective gate it never
had — **measure class separation before spending a bank; below about 0.04, no
amount of transplant work will help.**

### The original argument for B, kept as written before the run

**Redesigned after a correction.** The original version screened candidate rules
at four examples. That would not have separated the two explanations, because a
failure at four examples is expected for almost any category rule and says little
about the model.

**Procedure.** Two things vary, not one: the **number of demonstrations** (4, 8,
16, 32) and the **kind of rule** — one axis with two sides, like the injected task
that works, versus a category spanning unrelated members, like parity. Everything
written out in plain text, nothing patched, no site, no intervention.

**What each outcome means.**

| Outcome | Reading |
|---|---|
| Parity clears the bar at 16 or 32 examples | The branch was starved of demonstrations, not blocked. It reopens with a bigger budget, and five runs of machinery become usable |
| Category rules fail at every budget, axis rules pass at four | The interface can pick a side of an axis and cannot define a category, at any budget this model will take. Natural states do not come in two-sided pairs, so the branch is closed **for this interface** — and the honest next move is a different way of eliciting the report, not a bigger model |
| Everything passes at 32 | The reporting design has been running at the wrong budget all along, including the results already published, and those need re-examining before anything else |

The middle outcome is the one that would matter most, because it would mean five
failed runs were caused by a design choice made at the very start and never
questioned.

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
