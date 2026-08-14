# Empirical AI-safety portfolio for SPAR Fall 2026

I am an ML engineer moving into empirical AI-safety research.

The safety north star is causally source-faithful activation monitoring under
shortcut pressure. Efficiency is secondary.

## The result: models under-use what is plainly inside them

Give a language model a look at its own internal state and ask it what it sees.
Then give a **trivially simple outside method** the exact same internal state and
ask it the same question. The outsider wins, and it wins almost everywhere.

The outsider here is the cheapest thing that could possibly work: take the four
worked examples the model was shown, average the ones labelled `Q` and the ones
labelled `K`, and ask which average the fifth state sits closer to. Two averages
and a comparison.

| the task | model | two-average outsider | times the model won and the outsider lost |
|---|---:|---:|---:|
| tell a pushed-in idea from its opposite | 0.917 | **1.000** | **0** |
| tell two different pushed-in ideas apart | 0.899 | **1.000** | **0** |
| tell apart two meaningless directions | 0.663 | **1.000** | **0** |
| detect a very small nudge | 0.497 | 0.833 | 14 |

Across **1728 trials and four different task shapes, there are 14 where the model
answers correctly and the two-average comparison does not.** Guessing scores 0.500
everywhere in this table.

This matters because the standard test for whether a model is genuinely reading
its own internals — the one the field actually uses — asks whether the model beats
any outside method of equal or lower cost
([Song et al.](https://arxiv.org/abs/2508.14802)). By that test, this model does
not read its own internals at all. It reads them **worse than a stranger does**.

### Why the numbers can be trusted

**The strongest property is structural, not statistical.** Every trial has a twin:
the visible text is identical character for character, but the hidden state is
different and the correct answer is the opposite one. So anything working from the
visible text alone is stuck at exactly 0.500 — **not as a measured result, but by
construction.** The usual objection to this whole literature, that the model is
really just reading the prompt, cannot apply here.

The two do-nothing conditions land exactly where the design forces them: no edit
at all gives 0.5000, and editing only the query without the worked examples gives
0.5000.

## The model does read meaning, and finding that out took undoing my own result

Partway through, I audited my own concept bank and found something bad: the eight
"different" ideas I was pushing in all shared a large common ingredient. Every one
of the 28 pairs overlapped in the same direction. Worse, the simple reader I was so
impressed by turned out to **be** the average of those ideas — it matched at 0.99999.

So the task I thought measured *"which idea is this?"* actually measured
*"was something pushed in, and which way?"* A single yes/no axis. Not meaning.

The obvious next conclusion was that none of this was ever about meaning. **I
tested that instead of assuming it**, by changing the two options from *one idea
versus its opposite* to *two genuinely different ideas*, and changing nothing else:

| what the two options were | model scores |
|---|---:|
| one idea against its opposite | 0.917 |
| **two different ideas** | **0.899** |
| two meaningless directions, pushed equally hard | **0.594** |

Telling `garden` from `camera` is nearly as easy for the model as telling `garden`
from *not*-`garden`. Two meaningless directions, shoved in exactly as hard, are
much harder. So the model **is** reading meaning — and it still loses to the
two-average outsider, which is the finding above.

Both things are true at once, and neither is what I expected when I started.

## Earlier replication: a model can hold an idea after losing access to it

The model is `Qwen2.5-0.5B-Instruct`, a small open model with 24 layers.

While the model reads a short neutral text, I reach inside and add an idea to its
internal state. Then I remove the thing that added it, and check on every trial
that it is really gone. Only then do I invent a code (*ocean means Q, bread means
K*), paste it in, and ask which letter applies.

**The code did not exist when I planted the idea, so the planting cannot have been
aimed at the answer.** Eight ideas and eight letters, so guessing scores 0.125.

I chose how hard to push using one set of ideas, then froze that setting and ran a
completely separate set once. That second run is the table below.

| planted at layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| **use** (picks the right letter; guessing = 0.125) | **0.500** | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| **storage** (a simple reader finds the idea) | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |

**The bottom row never drops, so the idea is always still in there. The top row
falls to guessing by layer 14. What breaks is not memory. It is the model's ability
to reach what it kept and act on it.**

This is the same phenomenon as the headline table, reached by a completely
different route: a different model size, a different way of measuring, and a
design built for another purpose entirely. That is why I lead with it as one
finding rather than five.

### The obvious objection, and the check that answers it

If I add something to the model and a reader then finds it, maybe the reader is
only seeing the thing I added.

So I built a fake final state: the clean state plus exactly the same addition,
glued on with no processing in between. If the reader were only seeing my addition,
this should score as well as the real thing. It scores **0.167** against **1.000**.
The addition alone scores about guessing. So the model's own layers have to
transform it first into something resembling its ordinary idea of *ocean*.

There is one case where the reader *should* be fooled — plant and read in the same
place, nothing in between — and it comes out at 1.000 for both. The check can
detect the problem. It just does not find it anywhere else.

![propagation control](../activation-introspection/figures/retained_propagation.png)

### It is a replication

After building it, I checked the literature against what I had actually built. The
basic setup is Lindsey's, and the "only works if you plant it early" pattern is
already published for a larger model. What is left as mine: the invented-afterwards
code, the fake-state check above, and the run across three model sizes. Details in
[LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md).

## What training does, and the two claims I had to withdraw about it

Training a model to describe its own internals is the intervention the flagship
project proposes, so I ran it. Two findings survive, and both are narrower than
what I first wrote down.

**Training makes the model sensitive to smaller nudges, and widens what it can use
as a code.** At a nudge so small the untrained model is blind — exactly 0.500 —
trained versions reach 0.79–0.86. Trained versions also handle **arbitrary**
directions at 0.91–0.96, where the untrained model is at guessing. **Corrected
2026-08-14:** an earlier version of this line said the model "confidently reports
meaningless directions" and answers *"did something move in here?"* rather than
*"is idea X active?"*. That overstates it — those arbitrary directions are shown in
the worked examples before the question is asked, so there is a correct answer and
the model gives it. The finding is about the range of directions that can carry a
code, not about false alarms.

**What I withdrew, twice in one day.** I first concluded that training is just a
worse version of a simple statistical reader. Then I found the reader collapses on
meaningless directions while the trained model does not, and concluded that
training buys something no reader can have. **That second conclusion was wrong**,
and for an unglamorous reason: my reader was handicapped. It got one fixed rule
learned once, while the trained model got four fresh examples every single trial.
Run the reader the way the model is run — refitting on each trial's own examples —
and it scores **1.000** on exactly those meaningless directions.

So the generality belongs to adapting per trial, not to training. What survives is
that training lifts the model from blind to roughly **where the two-average
comparison already sat** — 0.833 — rather than past it.

I fixed one badly matched comparison and immediately made another one level up.
That is written down in [notes/13](../activation-introspection/notes/13-shared-axis-audit.md)
and [notes/15](../activation-introspection/notes/15-matched-reader-on-content.md)
with the wrong version left visible above the correction, because the mistake is
more useful than a clean story.

## Why one branch was stuck for five runs

Everything above pushes ideas *in*. The harder question is whether a model can
report on something it worked out **by itself**, and five attempts at that failed.

The blocker was not the surgery, which works, and not the reporting interface,
which works. It was the hidden rule I picked: *even answer → Q, odd answer → K*.
The model scores 0.533 on that **with the arithmetic written out in plain sight and
nothing hidden at all.**

I originally wrote that up as "the model cannot learn rules from four examples."
That was wrong — four examples is very few, and models are not good at learning
from few examples, so the failure said more about my setup than about the model.
The real explanation came from testing six rules with nothing hidden:

| the hidden rule | score | how tightly its two groups cluster |
|---|---:|---:|
| the answer was already shown (ceiling) | 0.979 | 0.218 |
| **animals versus tools** | **0.885** | 0.070 |
| numbers above versus below five | 0.729 | 0.074 |
| no rule at all (floor) | 0.490 | 0.008 |
| starts with a vowel versus a consonant | 0.479 | 0.000 |
| **odd versus even answer** | 0.469 | **−0.023** |

The model learns rules on brand-new examples perfectly well — *animals versus
tools* reaches 0.885 on words it has never been shown. What decides success is
whether the two groups **clump together** inside the model's own representations.
Every rule it learns has a clumping score between 0.043 and 0.218. Every rule it
fails is between −0.023 and 0.008. No overlap, at three different depths.

**Odd-versus-even is the only rule with a negative score**: `4+4, 2+2, 6+2` and
`3+4, 5+2, 1+2` share their numbers, symbols and length, so each sits *closer to
the opposite group* than to its own. It was close to the worst possible choice, and
five runs went into it.

That failure now hands the branch a cheap safeguard it never had: **check whether
your groups clump before spending an experiment on them.** About 150 quick passes,
no surgery required.

## The method, which is the point

I work out what a measurement really measures. I look for ways an experiment can
give a convincing answer for the wrong reason, in either direction. And when a
corrected comparison kills a result, I drop the result.

[CLAIMS.md](CLAIMS.md) grades every statement here, including the ones that did not
survive:

- A headline correlation of `-0.774`, retracted, because the two halves of the
  comparison were measured at different places in the model.
- A "100% identification" result, killed by my own control once I ran it.
- A training setup that scored 0.917 while putting essentially no probability on
  either answer — it had become a statistical reader wearing the model's voice, and
  no forced-choice score can detect that.
- Three "controls" that turned out to be arithmetic and could not have failed.
- A count of "28 out of 28" that was true by construction, because I took absolute
  values before counting. The conclusion survived being measured properly. The
  measurement that appeared to support it did not.

I keep the same ledger in unrelated work. My [ARC White-Box Estimation
Challenge](https://github.com/SkyeNygaard/AI-Safety-Roadmap) repository carries a
`claims.csv`, a research log, and the full record of what failed, next to a graded
competition entry and a proof. Different field, same discipline. Doing it once is a
habit. Doing it twice, independently, is a method, and the method is what I would
bring to a project.

## Where this reaches the two target projects

Original research is being done for these two. The other four applications are
written from work already finished — the reasoning is in
[RESEARCH-DIRECTION.md](RESEARCH-DIRECTION.md).

| SPAR project | Current fit | Why |
|---|---|---|
| [Introspection Training for Verbalization Activations](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz), Belinda Li | **The project's own first experiment, executed, with a result that constrains it** | The project proposes that supervision "comes cheaply from the internals themselves: probe readouts". I ran that comparison. The cheap readout is at least as good as the trained model everywhere I measured, and training moves the model toward it rather than past it. Training does buy a real change — sensitivity to nudges the untrained model cannot see — and pays for it by reporting meaningless nudges just as confidently. Two of my own conclusions about training were withdrawn on the way, both from badly matched comparisons |
| [Faithfulness, Self-Knowledge, and Introspection](https://www.sparai.org/projects/f26/rec3KQAI0JcxJJAce), Noah Siegel | **The project's central question, answered with a measurement rather than an argument** | Whether a self-report reflects genuine self-knowledge is exactly the gap measured above, on four task shapes with the visible text held identical so prompt-reading is impossible by construction. The answer is negative and precise: 14 trials in 1728. The interesting cases are the ones where the information is sitting in the model's internals, cleanly separable, and the model still gets it wrong |

Official resources: the [Fall 2026 project list](https://www.sparai.org/projects/f26/),
the [application advice](https://www.sparai.org/advice/), and the [mentee
application](https://forms.sparai.org/spar/mentee-app).

## What I would do next

**Reopen the self-computed-state branch, properly this time.** It stopped because
of a rule whose groups do not clump. A rule whose groups do clump works at 0.885
through the identical interface. But that rule was chosen by looking at results, so
it is a lead and not a finding: the next run has to fix the rule in advance, use a
fresh set of problems since both earlier ones are spent, keep the nothing-hidden
screen as a gate so a null can be read as a real null, and check each transplant
individually.

**Two things I would not do.** Not more training runs — the training arm has given
what it has to give, and its main lesson is now measured from two directions. Not
another model or another depth until the questions above are settled, because
robustness on an unsettled result is not worth buying.

## The rules I hold myself to

Written down here so they can be checked against what I actually published.

- Every claim says who it is about, what it measures, and what counts as one
  independent observation.
- Repeated prompts, orders, layers, and trials do not quietly get counted as
  independent repeats. Usually they are not.
- "The error bars overlap zero" is not evidence of no effect. A no-effect claim
  needs a threshold chosen in advance and a test against it.
- Safety numbers appear next to what they cost in usefulness, both measured on data
  the system did not see during tuning.
- Exploratory work stays labelled exploratory. Confirmatory tests use held-out data,
  once.
- Before an experiment runs, I write down what each possible outcome would mean —
  including the boring one, and including the outcome that would embarrass me. Every
  result in this repository has that note attached, written beforehand, with wrong
  predictions left in place.
- Any summary number has to be rebuildable from the raw per-item data, the settings,
  the prompts, the model version, and the environment. Older files that break this
  rule are marked invalid in [CLAIMS.md](CLAIMS.md) rather than quietly reused.

## Start here

- [CLAIMS.md](CLAIMS.md) lists every claim, what it actually measures, and whether
  it still holds. It includes the retracted ones.
- [RESEARCH-DIRECTION.md](RESEARCH-DIRECTION.md) says which projects get original
  research and why, what is established, and what is being tried next.
- [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) gives all six briefs and records where this
  work does not reach.
- [EXPERIMENTS.md](EXPERIMENTS.md) gives the studies I would run next, with their
  controls and stopping conditions.
- [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md) separates extension candidates
  from replications. **It was last checked 2026-08-09 and does not yet cover the
  results above**, which is the largest known gap in this repository.
- [AUDIT-MANIFEST.md](AUDIT-MANIFEST.md) gives the repository state and the commands
  to verify it.
- [AI_ASSISTANCE.md](../AI_ASSISTANCE.md) says how much was done with AI help, which
  is most of it, and what that does not license.

The lab record is [notes 01–16](../activation-introspection/notes/), in order. Each
note carries the reasoning written before the run, so the wrong predictions are
still visible next to the results that corrected them.

## The two code repositories

| Repository | What it shows now | What it does not show |
|---|---|---|
| [activation-introspection](../activation-introspection/) | Working PyTorch code for causal internal edits; the cost-criterion comparison across four task shapes; a meaning-versus-disturbance test; a bank audit that found a real flaw in my own setup; a retained-trace replication; and a correction trail | Reporting on self-computed states, trained free-form explanation, or any result on more than one model |
| [adaptive-monitor-sandbox](../adaptive-monitor-sandbox/) | A small world where an agent acts over many episodes and a monitor watches. Working plumbing, honest accounting of safety against usefulness, and a record of four separate ways the measurement was broken, with repairs implemented and tested | Real adaptive attacks by a real model, or replacement numbers from the repaired version |
| [spar-application](./) | The evidence map, the claim ledger, and the per-project briefs | A substitute for actually running the repaired experiments |

Read the code repositories as engineering evidence, an audit trail, and executed
experiments. The narrow claims that survive are scoped in [CLAIMS.md](CLAIMS.md),
and none should be stretched beyond the exact model, prompts, and setups they were
measured on.

## Scope, plainly

One model family. One depth for the main results. Everything pushed in from
outside rather than computed by the model itself. Four idea-pairs in the
meaning test. Three to four repeats where a proper interval would want more.

None of it shows a working safety monitor. What it shows is a set of controls
sharp enough to kill results that look like working safety monitors — including
four of my own.

## Publication state

Nothing here is published or peer-reviewed. The numbers are from single runs on one
machine, recorded with their settings, raw per-trial data, and model versions so
they can be checked. Every experiment's reasoning was written before it ran.

## License

MIT. See [LICENSE](LICENSE).
