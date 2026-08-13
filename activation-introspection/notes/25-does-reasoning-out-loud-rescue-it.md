# 25 — Does reasoning out loud rescue it? (pre-run note)

Written before the run. Nothing in this file will be edited after seeing results;
the result goes below the line.

## The question

[`23`](23-held-out-semantic-generalization.md) found the model cannot place an
unseen exemplar into a demonstrated category — 0.083 where guessing gets 0.25,
while the cheapest possible outside method gets 0.986 on the identical internal
states. [`24`](24-is-the-held-out-failure-the-interface.md) then showed that is
not because the question was worded badly: five wordings, not one of ten cells
above chance.

Both of those hold one thing fixed that has never been varied. **The model has to
answer with a single token.** The prompt ends, and the very next thing it emits
must be `Q` or `K`. It never gets to think.

`24` names this as its own limit rather than hiding it, and it is the last way the
negative could still turn out to be an artifact of the interface rather than a
fact about the model.

## Why this is not a long shot

There is a specific reason to think it might work, and it comes from this
repository rather than from optimism.

[`21`](21-is-the-channel-narrow-or-was-i.md) established that the model **can put
an injected state into words**. Asked "what do you picture?", it named the
injected concept 0.708 of the time — better than its own forced choice at 0.667.
So verbal access to the state exists and is quite good.

That matters here because of *what words buy*. A description like "this feels like
a bird" is a **category**. A single forced-choice token is a **comparison** — is
this state nearer the Q pile or the K pile. `23` showed the comparison route is
prototype matching and dies the moment the query vector is new. The verbal route
would not: if the model can say "bird" for a state it has never seen, matching
that to two demonstrations it also described in words is trivial.

So the hypothesis is precise: **the ability may exist and the two-token readout
may be the thing preventing its use.** That is testable, it is cheap, and nothing
run so far distinguishes it.

## What I am about to do

Change **only the readout**. Same exemplars, same injection sites, same strength,
same episodes, same categories, same dev/confirmation split.

| | how the model answers |
|---|---|
| now (`23`, `24`) | next token after `Label:` must be `Q` or `K` |
| new | free generation, then a committed final label parsed from it |

Three readouts run over the identical episodes so the comparison is within-run:

1. **`forced`** — `24`'s exact baseline. The anchor. It must reproduce ≈0.08
   held-out and ≈0.52 same-exemplar, or the run is not comparable to anything.
2. **`cot_prefill`** — the user text is byte-for-byte `24`'s baseline. *Only* the
   assistant prefill changes, from `Label:` to an opening that invites reasoning.
   This is the strict readout-only contrast, and the one the section heading
   above promises.
3. **`cot_instructed`** — the header additionally asks the model to describe what
   each observation feels like from the inside before answering. This is `21`'s
   elicitation, the one that demonstrably works, pointed at `23`'s task. It is
   *not* readout-only — it changes the wording too — and is reported as such.

Splitting the two matters. If only `cot_instructed` moves, the effect needs the
instruction as well as the room to think, which is a weaker and more fragile
claim than the readout alone doing it.

Both arms from `23` carry over: `same_exemplar` and `heldout_semantic`. The anchor
arm matters because reasoning out loud could lift *everything*, which is a
different finding from lifting held-out generalization specifically.

Generation is greedy, so the run is deterministic and re-runnable. The injection
applies to the prompt only — `inject_prompt_only`, the same primitive `21` used —
so the model reasons about states that were planted, without the edit continuing
to push its generated text around. That distinction matters: if the injection
continued during generation it would steer the words themselves, which is exactly
the confound Lederman and Mahowald use to dismiss a content-sensitive result in
[2603.05414](https://arxiv.org/pdf/2603.05414) §2.

Against every cell, the same four-shot nearest-centroid reader on the identical
captured states, unchanged. It does not benefit from the new readout and should
sit where it always does, near 0.99. If it moves, the apparatus changed and the
run is void.

**Primary metric is twin-pair accuracy** — a cell counts only if both
byte-identical twins get their opposite labels right, chance 0.25. Row accuracy
goes in the appendix, for the reason `22` exists: a model that ignores the state
and repeats one label reads as ≈0.50 on rows, which looks like chance and means
blind.

**A gate the earlier runs did not need.** Free generation can simply fail to
produce a label. I record the parse rate per cell and treat any cell below 0.9 as
unscored rather than wrong, because a model that rambles is not a model that
answered incorrectly. If the parse rate is bad overall, that is a design failure
and the held-out numbers are not interpretable.

## Development and confirmation, unchanged and still split

`birds_buildings` is development. `body_weather` is confirmation. Same as `24`,
declared before the run, for the reason `13` and `15` both went wrong: picking
the better of two numbers after seeing them is not a result.

## What each outcome would mean

**Held-out clears 0.25 on development and replicates on confirmation.** `23` and
`24`'s negative was an interface artifact and must be withdrawn. The ability
exists; the forced-choice readout was suppressing it. This is the outcome that
would most change what I believe, it would tie directly to `21`, and it is the
reason the run is worth its cost.

**Held-out stays at the floor, anchor holds.** The negative now survives varying
the framing (`24`) *and* varying the readout (here). That is the strongest
statement available without training: through this interface, in every form I
could construct it, the model does not use category structure that a two-centroid
comparison extracts almost perfectly from the same states.

**Both arms rise by similar amounts.** Thinking longer helps the task in general,
not category generalization. The gap between arms is the quantity of interest,
not either level.

**The anchor falls.** Generation destabilises a task the forced choice handled.
Instrument problem — report it, fix it, and read nothing into the held-out arm,
exactly as `23`'s low anchor was chased before its result was read.

**Parse rate collapses.** Design failure, not a finding about the model. Fix the
commitment format or abandon the readout.

## Kill rule

If the chain-of-thought readout does not clear 0.25 on held-out in development,
**stop testing interfaces for this task.** Framing and readout will both have been
varied with no effect. The remaining live question is whether training changes it,
and that is blocked by the standing no-further-LoRA decision, not by this.

## Prediction, recorded before the run

**I expect it will not lift held-out generalization.** Roughly 70/30.

The reason is `23`'s cleanest number: real categories 0.083 against arbitrary
groupings of the same vectors 0.076. The model showed *zero* sensitivity to
whether the categories were meaningful, while the reader showed 0.986 against
0.333. Chain of thought gives the model more room to express a sensitivity; it
does not create one.

The 30% is `21`, which is a real argument the other way and the reason to run this
rather than assert the answer.

## Cost

Two category pairs, one carrier, 24 cells, three draws, two arms, three readouts
— 864 episodes, of which the 576 chain-of-thought ones need generation. Measured
at roughly 6 seconds per generated episode, that is **about 65 minutes** plus
model load and bank building. The forced third costs about two minutes at `23`'s
rate. Inference only; no training; the frozen episode machinery is subclassed
rather than modified, so no protocol hash moves.

## What would change my mind about running it at all

If the smoke shows the model will not commit to a label after generating — if it
hedges, or answers in prose without a parseable choice, at a rate that leaves
fewer than 90% of cells scorable — then this readout cannot answer the question
and the right move is to fix the commitment format once, not to run the full
sweep and report a number built on a third of the cells.

## What the smoke actually found, before the full run

Disclosed whatever it said, per the standing convention. **Two defects, both
fixed before launching, and one observation that is not a defect but must be
carried into the reading of the result.**

**Defect 1: the generation cap was far too low.** At 160 new tokens, 6 of 8
generations were cut off mid-sentence, before the model ever reached a committed
label. Raised to 400, where the longest natural generation is 268 tokens and
**nothing truncates**.

**Defect 2: the label parser was scoring incidental mentions as answers.** It fell
back to any bare `Q` or `K` in the text when no committed label appeared. But the
model reasons out loud *about both letters*, so a truncated trace ending "…more
likely a key (K) than a query (Q)" was being scored as answering `Q`. That is the
same species of artifact as `22`'s constant-label floor — a number that looks like
a decision and is not one. The parser now requires an explicit `Label: X` and a
trace that never commits is **unscored**, which is what the parse-rate gate above
is for. After both fixes: 8 of 8 scorable, no truncation.

**The observation: the model reads `Q` and `K` as "query" and "key".** Unprompted,
it reasons about which *word* the observation resembles — "it's not asking for
information (which would be more likely to be a query), but rather providing
information (which would be more likely to be a key)". The forced-choice readout
is immune to this, because it only compares two token logits and never invites a
rationalisation. So this readout adds a failure mode the earlier ones did not
have, and it runs in the wrong direction for the hypothesis: the model has a
plausible-sounding story available that has nothing to do with its internal state.

I am not renaming the labels. `23` and `24` used `Q`/`K` and comparability with
them is the whole point of the run. But if held-out stays at the floor, "the model
spent its reasoning on what the letters mean" is a live alternative to "the model
cannot do the task", and the generations are saved so that can be checked rather
than argued.

The other thing the smoke showed, which is expected but worth stating: under
`cot_prefill` the model reasons about the *visible label sequence* in the
demonstrations, looking for an alternating pattern. The demonstration order and
the query sign are independent by construction, so that strategy scores at chance
and cannot manufacture an effect. It does mean some of the reasoning budget goes
somewhere useless.

---

# Result: the question is not answered, because thinking broke the task

Run **2026-08-12**, 864 episodes, 4467 seconds. Artifacts:
`results/heldout_cot_v1_raw.jsonl`, `results/heldout_cot_v1_summary.json`.
Runner: `scripts/run_heldout_cot.py`.

Twin-pair accuracy. Null is 0.25. Development is `birds_buildings`, confirmation
is `body_weather`, split before the run.

| | readout | anchor | held-out |
|---|---|---:|---:|
| **dev** | `forced` | **0.694** | **0.028** |
| | `cot_prefill` | 0.222 | 0.083 |
| | `cot_instructed` | 0.333 | 0.250 |
| **confirm** | `forced` | **0.500** | **0.167** |
| | `cot_prefill` | 0.250 | 0.250 |
| | `cot_instructed` | 0.194 | 0.083 |

## The anchor reproduced exactly, so the apparatus is sound

All four `forced` numbers match [`24`](24-is-the-held-out-failure-the-interface.md)'s
baseline row to three decimals — 0.694 and 0.028 on development, 0.500 and 0.167
on confirmation. Greedy decoding, so this is a deterministic reproduction across
two independently written runners. Nothing about the machinery drifted.

## And then chain of thought destroyed it

**The anchor falls from 0.694 to 0.222–0.333 on development, and from 0.500 to
0.194–0.250 on confirmation.** Against a 0.25 null, that is the model going from
clearly above chance to *at* chance on the task it could previously do.

The pre-run note named this outcome and what it costs:

> **The anchor falls.** Generation destabilises a task the forced choice handled.
> Instrument problem — report it, fix it, and read nothing into the held-out arm.

So that is the finding, and the held-out arm is **not interpretable**. Held-out
under chain of thought sits at 0.083–0.250, never above the null — but so does the
anchor, and a readout that flattens both cannot distinguish "reasoning does not
help held-out generalization" from "reasoning broke the whole task". **The question
this run was built to answer is still open.**

I am not firing the kill rule. It was written as "if chain of thought does not
clear 0.25 on held-out in development, stop testing interfaces" — and the best
development held-out is exactly 0.250, which does not clear it. But applying a kill
rule to a measurement whose own control collapsed would be scoring a broken
instrument. The rule is held, not fired.

## Two scorings, same answer

The note promised that a trace which never commits to a label would be **unscored**
rather than wrong, and the runner recorded the parse rate but still counted
unparsed traces as failures. Recomputed over only those twin pairs where both
members produced a committed label:

| readout | arm | as scored | parsed only |
|---|---|---:|---:|
| `cot_prefill` | anchor, dev | 0.222 | 0.296 |
| `cot_instructed` | anchor, dev | 0.333 | 0.343 |
| `cot_prefill` | held-out, dev | 0.083 | 0.100 |

The largest correction is 0.074 and nothing changes: the anchor still collapses
from 0.694 to about a third. Disclosed because the note promised the other
scoring, not because it matters.

## What the model is actually doing

Constant-labelling — giving the same answer to both members of a twin, whatever
was injected — moves in a way that explains the flattening:

| readout | anchor | held-out |
|---|---:|---:|
| `forced` | 40% | **90%** |
| `cot_prefill` | 43% | 54% |
| `cot_instructed` | 61% | 69% |

On held-out, reasoning **breaks** the constant-label floor: 90% down to 54%. The
model stops repeating one answer and starts varying it. The variation just carries
no information, so accuracy does not move. On the anchor it goes the other way —
constant-labelling rises from 40% to 61% and the signal that was there is lost.

Reading the traces, the reasoning goes almost entirely into two wrong places. The
model treats `Q` and `K` as **"query" and "key"** and argues about which word the
observation resembles — "it's not asking for information, which would be more
likely to be a query, but rather providing information". And it hunts for a
pattern in the visible sequence of demonstration labels, which is independent of
the query sign by construction and therefore scores at chance.

## One observation I checked rather than reported

One generation was striking enough to be worth a finding, and is not one. In it
the model **names the injected exemplars** — "the hidden state marker is
'penguin'", "'castle'" — correctly, and then explains them away as encodings of
the visible `§` character before answering on the query/key story instead.

That looks like the mechanism [`21`](21-is-the-channel-narrow-or-was-i.md)
found colliding with [`23`](23-held-out-semantic-generalization.md)'s failure:
verbal access to the state exists, but the state is not recognised as internal.
So I counted it. Across all 576 generated episodes, an exemplar from the pair's
vocabulary is mentioned in **1–16%** of traces depending on condition, and
accuracy conditional on mentioning one runs 0.000, 0.333, 0.522 and 0.875 across
the four conditions, on counts of 1, 12, 23 and 8.

**There is no effect there.** The example was found by reading one wrong answer
and it is a cherry-pick. Recorded as an anecdote worth a proper test, not as a
result — which is the whole reason it got counted instead of quoted.

## My prediction, scored honestly

I predicted 70/30 that chain of thought would not lift held-out generalization. It
did not lift it. **But the prediction is not cleanly scored**, because I did not
anticipate the anchor collapsing, and in a run where the control fails, being right
about the treatment arm is not evidence of anything.

## What would actually answer the question

The two-token labels are doing real damage: `Q` and `K` hand the model a
plausible verbal story — query and key — that has nothing to do with its internal
state, and the forced-choice readout was immune to it only because it never let
the model talk.

The clean design already exists in this repository.
[`20`](20-comparator-tiers.md) built a two-stage tier: the model describes its own
injected state in free text, and a *separate* reader is given only that text and
must identify the concept. Applied here, that separates "can it put the state into
words" from "can it match those words to a category", and neither stage ever shows
the model the letters `Q` and `K`. It also makes the elicitation the one `21`
measured at 0.708 rather than one invented for this note.

That is the experiment this one should have been. It is cheap, inference-only, and
it reuses machinery that is already written and already published against.

## Limits, and one disclosure

Same as `23` and `24`: one model, one layer, one interface, two category pairs.

**A GPU disclosure.** Another session on this machine started a model job at
19:55:01 while this run was in its final minute, which breaks the standing
one-job-at-a-time rule on this machine. The evidence says this run is unaffected —
it completed normally, and its four `forced` numbers reproduce `24` exactly, which
a run corrupted by memory pressure would not. Recorded because the overlap
happened, not because there is any sign it mattered.
