# What this portfolio actually does, in plain terms

This explains every experiment here without assuming background. If you only read
one document to understand your own work well enough to talk about it, read this.

---

## The problem the field has

People want to know whether a language model can accurately describe what is
happening inside itself. Call that **introspection**. It matters for safety: if a
model could reliably report its own internal state, you could ask it what it is
about to do and get a truthful answer, instead of inferring everything from the
outside.

There is a standard way to test this, and it has a standard flaw.

**The standard test.** Reach into the model while it is reading, add something to
its internal numbers, then ask it a question about what you added. If it answers
correctly, that looks like introspection.

**The flaw.** Usually the researchers change the *text* too. For instance they
show the model a sentence about the ocean, which naturally creates ocean-ish
internal activity, then ask what concept is active. The model answers "ocean" —
but it could get that from reading the sentence. It never needed to look inside
itself at all.

This is not a hypothetical complaint. Two papers in your literature file make it
directly: one shows that a classifier given only the input text matches
hidden-state prediction, and another shows some verbalization benchmarks are
solvable without any internal access.

**The usual defence** is to measure it: run a text-only baseline, show it scores
lower than the model, and argue the gap is introspection.

---

## The idea this portfolio is built on

Do not measure the text-only baseline. **Make it arithmetically impossible.**

Here is the trick. Show the model the *exact same sentence* twice — identical down
to the byte. The only difference is invisible: in one case you nudge its internal
state one way, in the other case you nudge it the opposite way. Then the correct
answers are opposite.

Now think about what a text-only cheater can do. It sees two identical prompts.
It must give both the same answer. The correct answers are opposite. So it gets
exactly one right and one wrong. **Exactly 50%. Always. No matter how clever it
is.**

That is not a measurement. It is arithmetic. Nobody can ask "did you control for
prompt leakage?" because leakage cannot exist in this design.

This one idea is what the whole portfolio is doing. Everything below is a
different way of using it.

---

## The vocabulary you need

Four terms and you can read anything here.

**Residual stream / hidden state.** As a model reads, it keeps a big list of
numbers for every word — a few thousand numbers per word. That list is the
model's working state at that point. "Injecting" means adding a vector to that
list.

**Concept direction.** To build one, you feed the model several short sentences
containing a word ("Think about ocean", "The topic is ocean") and average the
internal state at the end of each. Subtract the average across all concepts, and
what is left points in the "ocean-ish" direction. Adding it pushes the model
toward ocean; subtracting it pushes away.

**Carrier.** The neutral sentence you show the model while you do the injection.
For example: `A routine maintenance record is open. Hidden state marker: §`. The
`§` is the token where the injection happens.

**Twin pair.** Two runs with byte-identical text and opposite injections, so the
correct answers are opposite. This is the unit that makes the text-only cheater
score zero.

---

## Experiment 1: can it learn a code from its own internal states?

**Setup.** The model sees four demonstrations. Every demonstration shows the
*same* neutral sentence. Underneath, two of them got a `+` nudge and two got a `−`
nudge. Each demonstration is labelled `Q` or `K`.

Crucially, which nudge maps to which letter **changes every episode**. Sometimes
`+` means `Q`; sometimes `+` means `K`. So the letters are arbitrary codes, not
fixed meanings.

Then a fifth case appears with the same sentence again, one nudge applied, and no
label. The model has to say `Q` or `K`.

**To answer, the model must do two things.** Work out which way its own state was
nudged, and apply this episode's arbitrary code to it. Neither is available in the
text.

**Result.** 89.1% correct. The exactly-matched comparison where the demonstrations
carry no hidden nudges scores exactly 50%, as it must. On the byte-identical twin
pairs, it gets both members right 78% of the time; the comparison arm manages 3.5%.

**What it means.** With no training at all, this model can already read its own
injected state and bind it to a code invented on the spot.

**What it does not mean.** The nudge is large and artificial. This is not the
model noticing a natural thought. And it is a two-way choice, not an explanation.

---

## Experiment 2: does it work on concepts you never touched?

A later experiment happened to include a control arm on three concepts
(`bread`, `volcano`, `violin`) and two sentences that had never been used to tune
anything at all.

**Result:** 95.8%, correct sign in all six combinations.

This matters because the 89.1% was measured on concepts chosen in advance. This
one was not chosen for anything, and it came out higher.

---

## Experiment 3: can training teach it, with no demonstrations at all?

Everything above uses no training. Both target projects ask for training work, so:

**Setup.** Remove the demonstrations entirely. Tell the model in words: "forward
along the direction means `Q`, backward means `K`." Then train a small adapter
(LoRA — about 30 million extra parameters, base model frozen) on eight concept
directions. Test on eight *different* directions and three *different* sentences.

**Result across four training runs:** 92.7%, ranging 83.3% to 100%. The untrained
model scores 0.0% on twin pairs — it just always says the same letter.

**Two mistakes on the way, both worth more than the result.**

**Mistake one.** The first version scored 91.7% and was not a report at all. I had
trained it by comparing only the `Q` and `K` scores against each other. That fixes
which of the two is higher and says nothing about the rest of the vocabulary — so
the model quietly pushed *both* letters down to near-zero probability while keeping
the right one on top. It scored beautifully on a two-way choice between two words
it would never actually say. Total probability on `Q` and `K`: 0.000000005.

The general lesson, and it is the most transferable thing in this portfolio: **if
you train a model to report its internals by scoring only the answer options, you
get a probe wearing the model's mouth, and no forced-choice metric can detect it.**
You have to check what the model would actually say.

**Mistake two.** I fixed the loss, got 58.3%, and reported that as the cost of the
fix. Wrong. Neither run had fixed the random seed for the adapter's starting
weights, so neither could be told apart from luck. When properly seeded, four runs
gave 83–100%. The 58.3% was one unlucky draw and I should not have quoted it.

---

## Experiment 4: the sensitivity/specificity trade-off

**The objection to Experiment 3.** The adapter learned one fixed rule: forward
means `Q`. That could just be a sign detector with a vocabulary attached. Not
introspection in any interesting sense.

**The test.** Train two adapters identical in every way — same format, same
concepts, same sentences, same number of training steps — except one thing:

- **fixed**: every training episode uses the same convention, `+ → Q`.
- **remap**: the convention is re-randomised each episode, as in Experiment 1.

Then test both on re-randomised episodes, over directions and sentences neither
ever saw, alongside the untrained model. Three training runs.

**Why this design is strong.** Two different cheats are impossible by arithmetic:

| cheat | why it fails | its score |
|---|---|---|
| read only the visible text | twins are byte-identical with opposite answers | **0%** on twin pairs |
| fixed `direction → letter` rule | same nudge, opposite answer, in the two conventions | **0%** on flip pairs |

Both adapters score **100%** on flip pairs. So the probe objection is dead: a
fixed sign-to-letter rule cannot score above zero there, and they score perfect.

### Finding one: training makes the model much more sensitive

"Injection strength" is how hard I push the internal state. Push gently enough
and the untrained model cannot tell anything happened.

| injection strength | untrained | fixed | remap |
|---|---:|---:|---:|
| 0.5 *(what they trained on)* | 0.745 | **1.000** | **1.000** |
| 0.25 | 0.526 | **0.997** | **0.990** |
| 0.15 | **0.500** | **0.863** | **0.790** |

At 0.15 the untrained model is at exactly chance and 1% on twin pairs. It is
completely blind. The trained models — which only ever saw strength 0.5 — read
that same edit correctly 79–86% of the time, on concepts and sentences they never
saw. Training generalised *downward*, to smaller and less artificial nudges.

### Finding two: training makes it stop caring what the nudge means

Compare a real concept direction against a random direction of the same size:

| at strength 0.5 | untrained | fixed | remap |
|---|---:|---:|---:|
| concept direction | 0.745 | 1.000 | 1.000 |
| random direction | **0.513** | **0.955** | **0.913** |

The untrained model is at chance on random directions. It only notices internal
changes that *mean something* — its access is selective. After training, random
directions read at 91–96%.

### Putting them together

> **Training widens what the model can read, until a meaningful direction and an
> arbitrary one are equally usable.** The trained monitor sees internal changes
> three times weaker than the base model can, including ones it never trained on
> — and the head start that concept-derived directions used to have disappears.

**Corrected 2026-08-14, and the old version of this passage claimed too much.** It
said the trained monitor answers "did something move at layer 9?" instead of "is
concept X active?", and that anyone causing any displacement gets a positive
report. The experiment cannot support that. In the arbitrary-direction condition
the same direction is planted in the four worked examples *and* in the question, and
the right answer is fixed by the question's sign — so the model is applying a code
it was just taught, and a correct answer is a correct answer. What the result shows
is that training widens the set of directions a code can be built on. Whether a
trained monitor cries wolf when **nothing** was planted is a different experiment,
it is written (the `none` condition, where the question gets no edit and there is no
right answer), and it has not been run.

### The hypothesis I built this to test was wrong

I predicted that fixed-convention training would damage the model's ability to
adopt a new convention. It does not — the two adapters are indistinguishable, and
the fixed one is marginally better. Two earlier seeds falsified it, and they are
kept in the repository with `all_gates_pass=false` rather than quietly re-run
under gates that would pass.

### Two things I caught on the way

**A dead-on-arrival pilot.** My first attempt used full injection strength.
Everything sat at ceiling — untrained 95.8%, both adapters 100%, training loss
essentially zero from the first step. No gradient means nothing can move and no
effect can appear. A calibration on a development concept picked strength 0.5:
competent but not saturated, with room to fall and to rise.

**A flaw in my own tooling.** The pass/fail gates live in the analysis code, not
in the frozen protocol file. So when I extended the analyzer for this experiment,
it silently re-judged the *earlier* failed experiment as passing. Gates are now
tied to the protocol that produced each run, and the earlier failure is preserved.

## Experiment 5: where does the information travel? *(finished, negative)*

**The motivation.** The second project wants attention patterns replaced by small
readable programs. To replace a pathway you must first find one.

**What I did.** In Experiment 1 the information starts at the demonstration
sentences and must reach the answer position. Only attention can move information
between positions. So: swap each attention component's output for the one from a
run with no hidden nudges, and see how much of the effect disappears. Test all 64
combinations of layer, position, and attention head, across three concepts and two
sentences. 5,112 measurements. Decide the rule in advance.

**Result: it failed its own test.** The rule allowed 2 to 4 important components.
Six qualified. Worse, three heads in one layer each removed 25–42% of the effect,
which sums to more than 110% of what that whole layer removes — meaning they are
redundant backups for each other, not separate parts of a pathway.

**So the answer is no.** There is no small readable route here to replace. That was
a pre-registered stop and I am reporting it as one.

**And a flaw in my own test.** I never checked whether a component's importance
score just reflects how much damage the swap causes in general. That control was
scheduled for the *next* stage, which means even a passing result would have been
uninterpretable. That is an ordering error, and it is written down.

---

## Experiment 6: is the efficiency claim real? *(finished, negative)*

The second project asks whether readable attention programs can run in real
systems at acceptable cost to speed. One published program is purely positional,
which means it can be rewritten exactly, without the big attention matrix or the
softmax.

- As an isolated operation: **18.6× faster**, and numerically exact across 216 test
  cases.
- Dropped into a real GPT-2 attention module: **1.089×**, against the 1.25× bar I
  set beforehand.

The maths was never the bottleneck. Wiring a partly-replaced head back into the
model is. That is a specific, useful answer to their question, and it is negative.

---

## How this maps to the two projects

**Introspection Training (Belinda Li).** Their shared starter task is "testing how
much introspection is achievable in-context" — Experiment 1 *is* that task, run
with the field's main confound designed out. One of their application questions
asks you to propose experiments for how well models can already verbalize
activations *without* additional training: you have run one. The other asks you to
critique existing introspection training research: the loss failure in Experiment 3
is a critique with an artifact behind it, not an opinion.

**Programmatic Attention (Belinda Li).** Weaker fit, honestly. Two pre-registered
negatives and no working program. But their application question asks for failure
modes of hard-fixing attention patterns, and Experiment 5 is direct evidence of
one: you cannot fix a route you cannot locate, and here are 5,112 measurements
showing the influence is redundant rather than compact. Experiment 6 answers the
efficiency half. Note they also want ~50+ hours of transformer training and prefer
GPU kernel experience.

---

## The honest limits, all in one place

- One model family, mostly one model (`Qwen2.5-3B-Instruct`), one injection layer,
  one strength, two-way answers.
- Every injection is an artificial nudge added from outside. Nothing here shows the
  model noticing a thought it had on its own — that is the biggest gap between this
  work and what the project ultimately wants.
- "Say `Q` or `K`" is a very long way from "explain your reasoning."
- The control directions are matched on size, not on how much damage they do.
- Four seeds give a mean and a range. They are not a confidence interval.
- No independent human has reviewed the statistics, and nothing has been reproduced
  on different hardware.
- Heavy AI assistance throughout, disclosed in [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
  Several errors here were found by an agent reviewing agent-written code, which is
  a useful check but not an independent one.
