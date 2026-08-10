# Empirical AI-safety portfolio for SPAR Fall 2026

I am an ML engineer moving into empirical AI-safety research.

The safety north star is causally source-faithful activation monitoring under
shortcut pressure. Efficiency is secondary.

## New result: a model learned a hidden-state codebook from identical text

I gave `Qwen2.5-3B-Instruct` four examples with byte-identical visible
observations. The examples differed only through causal `+` or `−` residual-stream
edits. Each episode assigned those states to `Q` and `K` afresh, then asked for the
label of a held-out hidden state.

An initial held-out artifact was inspected and then retained when audit exposed a
small normalization mismatch, joint test-bank centering, and incomplete source
provenance. Without changing the model, layer, strength, labels, or gates, I froze
a repair-confirmation over eight fresh concept directions, three fixed carrier
strings (two new plus one anchor), and all 24 balanced demonstration-order ×
label-map × query-state cells. A two-cell post-freeze smoke caused no retuning;
its target scored 2/2, and the complete 576-row design was then run once.

| condition | accuracy | exact crossed-bootstrap 95% interval |
|---|---:|---:|
| no hidden edit | 0.500 | [0.500, 0.500] |
| query edit only | 0.500 | [0.500, 0.500] |
| random / coordinate-shuffled direction | 0.658 / 0.660 | [0.599, 0.717] / [0.575, 0.760] |
| **DEV-centered concept direction** | **0.891** | **[0.816, 0.995]** |

The target beats the exactly scale-matched query-only arm by +0.391 [0.316,
0.495], the strongest random/shuffled direction by +0.231 [0.137, 0.286], and
preserves 100% next-token label-format integrity. In byte-identical visible query
pairs with opposite hidden states, it gets both answers right 78.1% [63.2, 99.0]
of the time. The intervals exactly enumerate the chosen crossed-bootstrap distribution
over these fixed banks; they are not population intervals over concepts or prompts.

![causal hidden-state codebook](../activation-introspection/figures/causal_codebook_icl.png)

The defensible contribution is a **matched-visible, causal in-context
activation-labeling benchmark**. It tests one controlled instance of the official
project's zero-training question while eliminating the visible sentence-content
shortcut. It does not show privileged self-access, natural-state monitoring,
safety robustness, or programmatic attention. The full
accounting is in
[`notes/06-causal-codebook-icl.md`](../activation-introspection/notes/06-causal-codebook-icl.md).

## Earlier replication: a model can hold an idea after losing access to it

The model is `Qwen2.5-0.5B-Instruct`, a small open model with 24 layers.

While the model reads a short neutral text, I reach inside and add a concept to
its internal state. Then I remove the thing that added it, and check on every
trial that it is really gone. Only then do I invent a code (*ocean means Q, bread
means K*), paste it in, and ask which letter applies.

**The code did not exist when I planted the concept, so the planting cannot have
been aimed at the answer.** There are eight concepts and eight letters, so
guessing scores 1 in 8, or 0.125.

I chose how hard to push using one set of concepts, then froze that setting and
ran a completely separate set of concepts once. That second run is what the table
below reports. Choosing and testing on the same data is how you fool yourself, so
I did not.

| planted at layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| **use** (picks the right letter; guessing = 0.125) | **0.500** | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| **storage** (a simple reader finds the concept) | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |

The "reader" in the bottom row is a simple classifier trained only on ordinary
text. It never sees the experiment. It just knows what *ocean* normally looks like
inside this model, and I point it at the same internal state that produced the
answer in the top row.

**The bottom row never drops, so the concept is always still in there. The top row
falls to chance by layer 14. What breaks is not memory. It is the model's ability
to reach what it kept and act on it.**

### The obvious objection, and the check that answers it

If I add something to the model and a reader then finds it, maybe the reader is
only seeing the thing I added.

So I built a fake final state: the clean, untouched state plus exactly the same
addition, glued on at the end with no processing in between. If the reader were
only seeing my addition, this should score as well as the real thing.

It scores **0.167**, against **1.000** for the real one. The addition on its own
scores 0.125 to 0.375, which is about guessing. So the reader is not seeing the raw
addition. The model's own layers have to transform it first into something that
resembles its ordinary idea of *ocean*.

There is one case where the reader *should* be fooled: plant and read in the same
place, with nothing in between. That case comes out at 1.000 for both. The check
can detect the problem. It just does not find it anywhere else.

![propagation control](../activation-introspection/figures/retained_propagation.png)

For a project about training models to describe their own internals, the useful
version of this is: **what training would need to fix is the reaching, not the
keeping**, and the problem depends on where you look rather than being the same
everywhere.

### It is a replication

After building it, I checked the literature against what I had actually built. The
basic setup is Lindsey's. The "only works if you plant it early" pattern is already
published for a larger model.

What is left as mine: the invented-afterwards code, which closes a loophole this
repository previously fell into, plus the fake-state check above and the run across
three model sizes. Details in [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md).

## The method, which is the point

The retained-trace experiment below is a replication. The causal-codebook result
above is a targeted extension candidate. In both cases, the method is the point.

I work out what a measurement really measures. I look for the ways an experiment
can give a convincing answer for the wrong reason, in either direction. And when a
corrected comparison kills a result, I drop the result.

The record is [CLAIMS.md](CLAIMS.md), which grades every statement here, including
the ones that did not survive:

- A headline correlation of `-0.774`, retracted, because the two halves of the
  comparison were measured at different places in the model.
- A "100% identification" result, killed by my own control once I ran it.
- Three "controls" that turned out to be arithmetic. They could not have failed
  whatever the model did.
- A feature meant to randomize which feedback an agent received, which never
  randomized.

I keep the same ledger in unrelated work. My [ARC White-Box Estimation
Challenge](https://github.com/SkyeNygaard/AI-Safety-Roadmap) repository carries a
`claims.csv` with the status of each claim, a research log, and the full record of
what failed, next to a graded competition entry and a proof. Different field, same
discipline. Doing it once is a habit. Doing it twice, independently, is a method,
and the method is what I would bring to a project.

**Scope.** Two executed activation experiments: one causal extension and one
replication. Programmatic attention remains a concrete proposal, not a result.

## Start here

- [CLAIMS.md](CLAIMS.md) lists every claim, what it actually measures, and whether
  it still holds. It includes the retracted ones.
- [EXPERIMENTS.md](EXPERIMENTS.md) gives the studies I would run next, with their
  controls and the conditions under which I would stop.
- [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md) separates the new extension
  candidate from the retained-trace replication and the unbuilt attention bridge.
- [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) gives the direct next experiments and
  records where this work still does not reach.
- [AUDIT-MANIFEST.md](AUDIT-MANIFEST.md) gives the repository state, the commands
  to verify it, and what still blocks publication.
- [AI_ASSISTANCE.md](../AI_ASSISTANCE.md) says how much was done with AI help,
  which is most of it, and what that does not license.

## The two code repositories

| Repository | What it shows now | What it does not show |
|---|---|---|
| [activation-introspection](../activation-introspection/) | Working PyTorch code for causal residual interventions; a frozen causal-ICL repair-confirmation with complete raw rows; a retained-trace replication; and a correction trail | Privileged self-access, trained free-form explanation, J-space structure, or deployed programmatic attention |
| [adaptive-monitor-sandbox](../adaptive-monitor-sandbox/) | A small world where an agent acts over many episodes and a monitor watches. Working monitor and feedback plumbing, honest accounting of safety against usefulness, and a useful record of four separate ways the measurement was broken. The repairs are implemented and tested | Real adaptive attacks by a real model, a trustworthy negative result, or any replacement numbers from the repaired version |
| [spar-application](./) | The evidence map, the repair plan, and the per-project briefs | A substitute for actually running the repaired experiments |

Read the two code repositories as engineering evidence, an audit trail, and two
executed activation experiments. The narrow claims that survive are scoped in
[CLAIMS.md](CLAIMS.md), and none should be stretched beyond the exact model,
prompts, and setups they were measured on.

## The checks behind the headline

Everything here stands behind the table above, including the parts that do not
hold up.

**Checks that passed.** If I just write the concept in plain text, the model scores
0.875, so it can do the task and a zero at depth means something real. A control
that scrambles the edit lands at 0.125 to 0.146, which is chance. The effect
survives if I throw out every trial where the model formatted its answer badly
(0.435 at layer 2). Six of the eight concepts individually beat twice chance. And
for layers 18 and 22 I did not merely fail to find an effect. I decided in advance
what "no meaningful effect" would mean (within 0.05 of chance) and those layers
fall inside it, which is a positive finding rather than an absence.

**Checks I withdrew.** Two "controls" read exactly 0.125 everywhere, and I once
reported that as passing. It is not a result, it is arithmetic. Each model run gets
scored against all eight concepts, and the codes cycle, so exactly one of every
eight rows is correct no matter what the model does. That happened in 144 cases out
of 144. Those checks confirm the wiring works. They could never have failed. The
scrambled-edit control is the one that could have come out badly, and the effect
survives it.

**Where the comparison is not perfectly fair.** Ideally the real edit and the
control edits would disturb the model by the same amount. At the layer-2 headline
they do not: the real edit disturbs it slightly more (1.50 against 1.00 and 1.37,
measured as how much the model's output distribution shifts). That is the same
direction as a result I disqualify below, though the gap here is much smaller. It
also runs the other way on formatting, where the controls break the answer format
*more* than the real edit does (0.44 against 0.21 and 0.33). So I do not think it
explains the result. But the edits are matched on size, not on how much damage they
do, and that is a real limitation.

**Bigger models.** Running the same thing at 1.5B and 3B, the point where the model
stops being able to use the idea moves **later** as models get bigger. The 3B model
is still at 0.354 where both smaller ones are at chance. All three are finished by
about three quarters of the way through the network. These runs are exploratory,
because I froze the setting on the smallest model and carried it over without
retuning, so the comparisons are not perfectly matched across sizes.

Within the retained-trace scale ladder, layer 21 is the sturdiest cell. Both control edits
disturb the model *more* than the real concept does, and both still sit at chance
while the real one reaches 0.354. That is exactly the comparison the "your edit was
just more disruptive" objection needs, and it survives. Layer 21 is the only case
that qualifies. At layer 15 only one of the two controls is above the real edit, and
the layer-3 case fails badly (2.44 against 0.51 and 0.41) and should not be quoted.

## Where this reaches the two target projects

| SPAR project | Current fit | Why |
|---|---|---|
| [Introspection Training for Verbalization Activations](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz), Belinda Li | **Executed in-context benchmark plus an executed training result** | The causal codebook tests the project's zero-training starter question while making the visible sentence-content shortcut impossible by construction, then replicates on three out-of-bank concepts at 0.958. A trained LoRA reporter then verbalizes the sign of an injected state at 0.583 on eight withheld directions, against 0.000 untrained. Its first version scored 0.917 while putting 5e-9 probability on the answer tokens, which is a trap this project would hit. Neither result establishes a robust safety monitor, and the trained arm is one seed. |
| [Deploying Programmatic Attention in Real Transformers](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L), Belinda Li | **Two executed negatives against frozen thresholds** | The disjoint head screen ran and stopped: influence is redundant across six components where the protocol allowed 2–4, so there is no compact route to program. Separately, an exact `O(TD)` lowering of one released GPT-2 program is 18.63× faster in isolation but 1.089× once integrated, missing a preregistered 1.25×, which localizes the cost to partial-head projection and dispatch. Neither is a circuit or a monitoring-robustness result. |

Official resources: the [Fall 2026 project list](https://www.sparai.org/projects/f26/),
the [application advice](https://www.sparai.org/advice/), and the [mentee
application](https://forms.sparai.org/spar/mentee-app).

## What I would do first

That staged safety study has had its first two stages run, and the second one
stopped it: influence over the reporting behavior is redundant across six
attention components, not compact enough to replace with a readable program. So
the thing I would do first is no longer the head screen.

The trained reporter is now run, and what it needs next is obvious from its own
limits: **independent training seeds**, because one LoRA run cannot separate the
effect from initialization luck, and then episode-remapped training so the
trained and in-context designs become comparable. Both are cheap. Neither
depends on a compact attention route existing, which is why this is the line to
push rather than re-running the head screen.

## What I would lead with

1. **I can build the thing and run the experiment.** Code that reaches into a
   running transformer, two-stage memory injection, local model scoring, persistent
   environments, monitors, metrics, tests, and two experiments carried through end
   to end, with settings frozen beforehand, every rerun disclosed, and every raw
   trial saved.
2. **I find my own mistakes.** I compared two things measured at different places
   and got a reversed correlation. I counted shuffled menu orders as if they were
   independent runs. I built a feedback channel that leaked the answer through a
   status code. And I let leftover state make an agent look more useful than it was.
3. **I retract instead of defending a good story.** "Decodability is not usability"
   and "no model up to 3B adapts" were both headlines of mine. Neither is a claim I
   would carry into an application.
4. **I check whether my idea is new, against what I actually built, and downgrade
   it when the answer is no.** This experiment's setup turned out to be prior work
   and its pattern already published. It is labelled a replication in every
   document, rather than quietly presented as new.
5. **I design around what the number really means.** What counts as one independent
   observation, matching the two sides of a comparison, tuning on one set and
   testing on another, and deciding in advance what "no effect" would look like.
   All settled before looking at the data, not after a result shows up.

## The rules I hold myself to

Written down here so they can be checked against what I actually published.

- Every claim says who it is about, what it measures, and what counts as one
  independent observation.
- Repeated prompts, menu orders, layers, and episodes do not quietly get counted as
  independent repeats. Usually they are not.
- "The error bars overlap zero" is not evidence of no effect. A no-effect claim
  needs a threshold chosen in advance and a test against it.
- Safety numbers appear next to what they cost in usefulness, both measured on data
  the system did not see during tuning.
- Exploratory plots stay labelled exploratory. Confirmatory tests use held-out data,
  once.
- Any summary number has to be rebuildable from the raw per-item data, the settings,
  the prompts, the model version, and the environment. Older files that break this
  rule are marked invalid in [CLAIMS.md](CLAIMS.md) rather than quietly reused.

## Publication state

Resolved on 2026-08-05. All three folders now live in one repository, so every
cross-reference works. That is 112 relative links, checked. The two code repositories were
joined with `git subtree`, so each keeps its own history. That history shows the
corrections happening as they happened, which is part of the evidence.

Both test suites pass from a fresh copy. They did not at first. The virtual
environments still pointed at the folder paths from before the merge, and
`adaptive-monitor-sandbox` only type-checked on machines where an optional
dependency happened to be installed. Both are fixed. The fact that this paragraph
was wrong when first written is the reason it was worth checking rather than
asserting. The one results file that has no reliable provenance is still excluded
from the repository. See [AUDIT-MANIFEST.md](AUDIT-MANIFEST.md).

## License

MIT. See [LICENSE](LICENSE).
