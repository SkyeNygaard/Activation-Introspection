# Pre-run note: is there any depth where the lens loses?

Written **2026-08-12, before the run.**

## What this follows from

[`17`](17-supervision-is-the-hidden-knob.md) argued that the field's cost
criterion cannot be satisfied wherever a state is linearly decodable, because the
comparison class contains the model's own unembedding — cheaper than 27 further
blocks and available to anyone with activation access. It ended with the design
principle: **the criterion is only informative on states a lens cannot read**, and
noted that no run here has ever checked which states those are.

This checks. It is the direct test of that note's own claim, and the cheapest one
available.

## What I am about to do

Inject a concept at block 9 exactly as `17` did, then read the state through the
final norm and unembedding at **every block from 9 to 35**, at the injection
position and at the final answer position. One forward pass per episode captures
all of them.

The model's number is already fixed at **0.597** overall and **0.667** at the
strength used here, from `17`. Nothing about the model changes; only where the
third party reads.

## The question, stated as a single number

**Is there any read depth where the lens falls below the model?**

If yes, that depth is the first place in this repository where the criterion is
satisfiable, and possibly satisfied — the regime `17` says the field should be
testing and nobody is.

If no, then at this site the criterion is unsatisfiable at every depth, and `17`'s
argument stops being an inference about one measurement and becomes a measured
property of the whole stack.

## What each outcome means

| Outcome | Reading |
|---|---|
| Lens dips below the model at some depth | **The finding.** Report that depth, then re-run the model's identification reading *only* from states at that depth. This would be the first satisfiable regime found here |
| Lens stays above the model at all 27 depths | `17`'s argument is measured rather than argued. Strong negative about the concept-injection paradigm: there is nowhere at this site to run a fair test |
| Lens collapses everywhere including the injection site | Harness fault — `17` measured 0.986 at block 9 and that must reproduce. Stop and fix |

## Prediction

The lens will stay above the model at every depth. [`11`](11-matched-cost-reader.md)
found a labelled centroid reader perfect across 25 consecutive blocks, and the
planted vector is a token-contrast construction, so it should stay token-aligned
for most of the stack. I expect degradation only in the last few blocks, and by
then the marker position is producing its own next-token prediction rather than
storing the concept — the error `11` disclosed in its own depth sweep.

Saying this in advance because a null here is the informative outcome and I do not
want it reframed afterwards as a disappointment.

## What it costs

24 episodes, one forward pass each, capturing 27 blocks per pass. Under a minute.
No training, no new banks.

---

# Result: the lens does dip below the model — and the criterion still fails, for a reason worth naming

Run **2026-08-12**. 24 episodes, 27 depths, two sites. Artifacts:
`results/lens_depth_v1_raw.jsonl`, `results/lens_depth_v1_summary.json`. Runner:
`scripts/run_lens_depth.py`. Model at this strength: **0.667**. Chance: **0.125**.

| block | lens @ marker | lens @ answer position |
|---:|---:|---:|
| 9–11 | **1.000** | 0.125 |
| 12 | 0.708 | 0.125 |
| **13–17** | **0.625** | 0.125 |
| 18 | 0.750 | 0.167 |
| **19–22** | **0.625–0.667** | 0.125–0.208 |
| 23–27 | 0.708–0.875 | 0.125–0.250 |
| 28 | 0.708 | 0.583 |
| **29–31** | **0.542–0.667** | 0.542–0.667 |
| 32–35 | 0.833–**0.958** | 0.667–0.792 |

## Legibility is U-shaped, which I did not predict

I predicted the lens would stay above the model everywhere and degrade only in the
last few blocks. **Wrong on both halves.**

It is perfect for three blocks, **falls to 0.625 through the middle of the
network — below the model's 0.667 — and recovers to 0.958 at the end.** The
planted vector gets digested into something less token-aligned as the model
processes it, then re-emerges in token space near the output head.

So there are depths where the model beats a lens read of its own state: blocks
13–17, 20–21, 29–30. **That is the first place in this repository where a cheap
reader loses.**

## And the criterion still fails, because the third party gets to choose

The criterion asks whether *some* equal-or-lower-cost process beats the model. The
third party is not obliged to read at a bad depth. It reads at block 9 and scores
**1.000**. Frozen verdict: `criterion_unsatisfiable_at_every_depth`.

**But the reason is now located precisely, and it is not about the model.** Block 9
is where *I* planted the vector. The third party wins by reading the experimenter's
own edit before the model has done anything to it. That is not a third-party read
of a model's internal state — it is a read of the intervention.

> **Any concept-injection design hands the third party a perfect read at the
> injection site, by construction.** The leak is the paradigm, not the model. A
> criterion evaluated against a reader with access to that site cannot say anything
> about introspection.

That is [`17`](17-supervision-is-the-hidden-knob.md)'s argument made specific and
measured, and it comes with a concrete repair: **bar the third party from the
injection site**, or better, test on states nobody planted.

## The answer position is the finding I did not go looking for

Read the column again. At the position the model actually answers from, the
concept is at **chance — 0.125 — for blocks 9 through 27**, nineteen consecutive
blocks, while the model identifies it at 0.667.

The information reaches the answer position, because the model uses it. It is not
*in token space* when it gets there. Whatever routes a concept from the marker to
the answer is not a copy a lens can follow, and it stays illegible for three
quarters of the stack before surfacing at block 28.

**This is the first result here that looks like the model doing work a cheap reader
cannot follow.** It is not privileged access — the model still had the marker
available, and so did the lens — but it is a mechanism the lens is blind to, and it
is exactly the kind of state `17` says the criterion should be tested on.

## Epistemic status

- **Observed:** every number in the table, 24 episodes per cell, one model, one
  injection site, one strength.
- **Inferred:** that the injection site is a leak by construction. This follows
  from the block 9–11 perfection plus the fact that block 9 is where the edit was
  applied, but "by construction" is an argument about designs, not a measurement of
  other people's designs.
- **Not established:** anything about privileged access. The mid-band dip shows the
  lens losing at *some* depths, not the model winning at the depths that count.

## Limits

- 24 episodes per cell, so 0.625 against 0.667 is one or two episodes. **The dip's
  existence is clear across ten consecutive depths; its exact size is not.**
- The lens uses the model's own unembedding, which is the point being made rather
  than a flaw, but it should be restated wherever these numbers are quoted.
- The model's 0.667 comes from a full forward pass with the marker available. It is
  not restricted to the answer position, so the last section is a statement about
  what the lens can see, not about what the model uses.
- One strength. `17` found the model at 0.458 at strength 1.0, which would sit
  below more of the lens curve.

## What this cannot do

It only sweeps *where* the third party reads, not *what kind* of state is being
read. Even a clean null leaves open the regime that matters most — states the
model computed for itself, which is where `17`'s argument says the criterion
should finally be informative, and which this design does not touch.
