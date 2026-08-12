# 23 — Held-out semantic generalization (pre-run note)

Written before the run. Nothing in this file was edited after seeing results.

## The question

`14` found the model distinguishes two concept-derived directions far better than
two random ones, and called that **content**. `15` confirmed the gap survives a
separation-matched random control.

But in both, the *same two vectors* appear in the demonstrations and in the query.
`eagle` is injected at the demo positions and `eagle` is injected at the query.
So the model never has to know what a bird is. It only has to notice that the
query state sits near the state labelled `Q`.

That is **prototype matching**, and it would produce every number `14` and `15`
report. "Content" is stronger language than the design licenses.

## What I am about to do

Keep the four-shot episode structure exactly as it is — 24 cells of six
demonstration orders, two label mappings, two query signs, all visible text
byte-identical — and change one thing: **every injection position gets a
different exemplar.**

| | demonstrations (+1 / −1) | query |
|---|---|---|
| now (`14`) | `eagle`, `eagle` / `library`, `library` | `eagle` or `library` |
| new | `robin`, `sparrow` / `museum`, `cathedral` | **`falcon`** or **`warehouse`** |

The query exemplar never appears in the demonstrations. To answer, the model must
place a state it has not seen into a category defined by two examples. Prototype
matching on the exact vector is no longer available.

Five arms:

1. **`same_exemplar`** — the old design, unchanged. An anchor. If this does not
   reproduce ≈0.80 twin-pair accuracy the apparatus is broken and nothing else in
   the run means anything.
2. **`heldout_semantic`** — the design above.
3. **`heldout_arbitrary`** — identical structure, but the two "categories" are
   arbitrary groupings of unrelated concepts, with the distance between the two
   class centres scaled to equal the semantic arm's. Same geometry, no meaning.
4. **`heldout_random`** — exemplars are random directions, magnitude matched.
   Floor.
5. **`query_only`** and **`clean`** — the existing leak checks, carried over
   unchanged.

Against every arm I run the **four-shot nearest-centroid reader** on the identical
hidden states. `15` and `22` both show that skipping this produces a claim that
does not survive one. Here the reader has to generalize too, so for the first time
in this repository it is not automatically at ceiling.

**Primary metric is twin-pair accuracy** — a cell counts only if both
byte-identical twins get their opposite labels right. Single-prompt accuracy goes
in an appendix. `22` is the reason: at the row level a model that ignores the
state entirely and repeats one label reads as 0.497, which looks exactly like
chance and was reported as chance.

## The gate that runs first

Before any of this, a cheap geometry check with no generation at all: build the
candidate exemplar vectors and measure whether exemplars of one category sit
closer to each other than to the other category's, at layer 9.

This is the capacity check. If the categories do not cluster at the injection
site, then the information needed to generalize is not present in the state, and
a failure downstream would say nothing about the model's reporting — only that I
injected structure the site does not carry. Categories are chosen on this gate and
frozen before the behavioural run, and the arbitrary arm is matched to whatever
separation the winning categories turn out to have.

Selecting categories on geometry and then reporting generalization is selection on
something correlated with the outcome. That is why the arbitrary arm is matched on
between-class distance: it is the same selection applied to structure without
meaning. `19` established clustering predicts learnability here, so this is the
established prospective use, and it is disclosed rather than buried.

## What each outcome would mean

**Semantic generalizes, arbitrary does not.** The model is using category
structure, not vector proximity. This is the result that would justify the word
content, and it is the only outcome that upgrades `14` rather than qualifying it.

**Both generalize.** The mechanism is geometric classification. Whatever the model
is doing works on any sufficiently separated partition, and "semantic" is simply
the wrong word for it. Not a null — it says the task is easier than `14` implied.

**Neither generalizes, anchor holds at ≈0.80.** `14`'s result is prototype
matching, pair-specific, and the content claim has to be withdrawn. This is a
clean, publishable negative and it closes a branch the handoff currently ranks
second.

**Anchor fails too.** Instrument failure. Fix the apparatus; report nothing about
introspection.

**The reader beats the model in every arm.** Expected, and it does not spoil the
experiment. This run asks what *structure* supports the task, which is a question
about the representation and not about privileged access. A reader that
generalizes semantically while the model does not is itself informative: it would
show the information is there and the model is not using it.

## Kill rule

If held-out semantic generalization fails on two independently chosen category
pairs while the anchor holds and the geometry gate passes, stop calling `14`'s
result semantic abstraction, amend the README, and do not run a third pair.

## Cost

Geometry gate: a few hundred forward passes, under two minutes, no generation.
Behavioural run: about 720 episodes at the ≈0.30 s/episode `15` measured, so
roughly four to six minutes, plus model load. Two category pairs, one model
(`qwen-3b`, layer 9, the frozen site), no training.

This is cheap because it reuses the frozen episode machinery untouched. The only
new logic is *which vector goes at which position* — one function.

## What would change my mind about running it at all

If the geometry gate fails for every candidate category pair, this design is
untestable at layer 9 and the right move is to ask whether any layer carries
category structure before spending the behavioural run.

---

# Result: it is prototype matching, and the model never sees the category

Run **2026-08-12**. Gate 254 s, behavioural run 1440 episodes. Artifacts:
`results/category_geometry_v1_summary.json`,
`results/heldout_semantic_v1_raw.jsonl`,
`results/heldout_semantic_v1_summary.json`,
`results/intervention_equivalence_v1_summary.json`.

## The gate passed easily, so the structure is there

| pair | held-out nearest-centroid | scrambled null, 99th pct | |
|---|---:|---:|---|
| `birds_buildings` | **1.000** | 0.704 | pass |
| `body_weather` | **0.989** | 0.747 | pass |
| `fruit_tools` | 0.965 | 0.681 | pass |
| `mammals_vehicles` | 0.931 | 0.722 | pass |

The scrambled null sits at 0.500 on the mean, so the calibration is honest. Layer
9 carries the category cleanly: an unseen exemplar lands with its own kind
essentially every time. Frozen choice, top two: `birds_buildings`, `body_weather`.

## The headline

| arm | **twin pair: model** | row: model | twin pair: reader | constant-label cells |
|---|---:|---:|---:|---:|
| `same_exemplar` | **0.521** | 0.760 | 1.000 | 48% |
| `heldout_semantic` | **0.083** | 0.542 | **0.986** | 92% |
| `heldout_scrambled` | 0.076 | 0.521 | 0.333 | 89% |
| `heldout_random` | 0.014 | 0.493 | 0.076 | 96% |
| `query_only` | 0.014 | 0.500 | 0.000 | 97% |

Twin-pair chance is 0.25. A model that ignores the state and repeats one label
scores 0.000.

**Hold the exemplars and the strength fixed, and move only the query vector out of
the demonstrations: the model falls from 0.521 to 0.083. The reader does not move
at all, 1.000 to 0.986.**

At 0.083 the model is below the coin-flip null, and 92% of its twin cells emit the
same label for both query signs. That is the constant-label floor from
[`22`](22-the-weak-arm-was-a-floor-not-a-frontier.md), and it is why the row
number is 0.542 — which reads as chance and means blind. Paired episode by
episode: 131 cells the reader gets and the model misses, against **1** the other
way.

## The comparison that settles it

`heldout_semantic` 0.083 against `heldout_scrambled` 0.076. **The model gets no
benefit whatsoever from the categories being real.** The reader, on the identical
states, goes 0.986 against 0.333.

So the semantic structure is present, it is load-bearing, and the cheapest
possible reader exploits it almost perfectly — and the model shows no trace of
using it. This is not a capacity failure. The information is right there.

Both frozen pairs agree: `birds_buildings` 0.014, `body_weather` 0.153, against
their anchors of 0.583 and 0.458.

## The anchor came in low, and that had to be chased before anything else

`same_exemplar` scored 0.521 where `14` reported 0.799. The pre-run note says a
failed anchor means the apparatus is broken and nothing else counts, so this was
checked before the result above was read.

`scripts/diagnose_intervention_equivalence.py` runs both intervention builders
over the same concepts, episodes and strength. **Agreement 96 of 96 episodes, both
builders at 0.7917**, reproducing `14`'s 0.799 on `14`'s bank. The new
per-position builder is the same edit as the old one.

So the shortfall is the words, not the machinery: these exemplars are simply
harder to report. It varies by exemplar and not by anything else —
`penguin`/`castle` reaches 0.833, `spine`/`monsoon` sits at 0.250. Since every arm
here uses the same exemplars at the same strength, the within-run comparison is
unaffected, and the anchor still clears both the held-out arms and the 0.25 null
by a wide margin.

## What this closes

`14`'s "it is content" survives as stated — two different concepts *are*
discriminated far better than two random ones. What dies is the reading that this
shows the model recovering **semantic structure**. It recovers a vector it has
already been shown. Move the vector and the ability goes, while the category it
belongs to stays perfectly legible to a four-shot centroid.

Per the kill rule: two independently chosen category pairs failed with the gate
passing and the anchor holding. **Stop calling `14`'s result semantic
abstraction. Do not run a third pair.**

## Limits

One model, one layer, one interface. The elicitation is forced choice over two
opaque labels, and [`21`](21-is-the-channel-narrow-or-was-i.md) showed wording can
move a reporting number from 0.29 to 0.71 — so this bounds what *this* interface
recovers, not what the model could be got to say. The honest scope: at layer 9,
through the four-shot opaque-label interface, this model does not place an unseen
state into a demonstrated category.

What would change my mind: the same design under an elicitation sweep, or after
introspection training, showing held-out generalization appear. That is a real
experiment and it is now the cheapest live descendant of this branch — the
apparatus, the gate and the controls all already exist.
