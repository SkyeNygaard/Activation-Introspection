# 33 — Three ways to boost introspection, one control (pre-run note)

Written before the run. The motivation section below was written before
[`32`](32-does-boosting-introspection-cost-selectivity.md) reported, so its result
does not shape this design.

## The programme this completes

Three separate literatures each report a way to make a model introspect better:

| boost | who | effect |
|---|---|---|
| **training** | [`08`](08-sensitivity-specificity-tradeoff.md), Introspection Adapters, IFT | detection floor extended to edits the base model is blind to |
| **refusal ablation** | [Macar et al.](https://arxiv.org/abs/2603.21396) | detection 10.8% → 63.8% |
| **prompting** | [Latent Introspection](https://arxiv.org/html/2602.20031) | detection 0.3% → 39.9%, ceiling 84.0% |

**Every one of them audits false positives the same way: against trials where
nothing was injected.** Searched again on 2026-08-13; that is still true of all
three.

The harder control has existed since Lindsey's original — **norm-matched random
directions**, where something real happened that meant nothing — and base models
pass it. `08` showed training fails it badly. `32` asks it of ablation. **Nobody
has asked it of prompting**, and nobody has asked all three the same question in
one design.

## What this run adds that `32` does not

Two things.

1. **A prompting arm.** The third boost, never audited this way.
2. **Confidence margins**, which `32` could not record because it reuses
   `score_pair` from the frozen `run_content_vs_disturbance`, and that returns
   correctness without a margin. Without margins there is no abstention curve, and
   [`29`](29-can-abstention-recover-selectivity.md) and
   [`31`](31-why-training-inverts-abstention.md) are the two most useful results
   here. So the scoring is re-implemented locally, identically apart from also
   returning `abs(logit gap)`, and the frozen function is left alone.

## Amendment after `32` reported, before this ran

`32` came back and it changes which arm matters. **Refusal ablation is not a boost
at this scale** — the direction was verifiably removed (refusal 1.00 → 0.00) and
the task got *worse*, content twin-pair 0.792 → 0.604. You cannot audit the price
of a boost that did not happen.

So the ablation arm here is demoted to a carried control — it is already
implemented, it costs nothing to include, and a second measurement of a null is
worth having. **The prompting arm becomes the experiment.**

That is the right focus on the evidence: [`24`](24-is-the-held-out-failure-the-interface.md)
already established that prompting is a boost that *works at this scale*, lifting
the anchor from 0.694 to 0.875 and cutting constant-labelling from 40% to 25%.
It is the only one of the three boosts demonstrated to transfer here, and it is
the one nobody has audited against a random-direction control.

## What I am about to do

`14`'s design unchanged — two concepts against two random directions at **identical
class separation by construction**, byte-identical visible text, twin-pair
scoring — crossed with three boost conditions:

- **`none`** — the published baseline. Must reproduce `14` (content 0.899 row) or
  the run is not comparable.
- **`prompt`** — an instruction that tells the model an edit may have been made to
  its internal state and to attend to it. This is the family Latent Introspection
  found moves detection most, and the family `24` found lifts the anchor here.
- **`ablate`** — the refusal direction removed at every layer, exactly as `32`
  builds it. Gate carried over: refusal on held-out harmful prompts must drop.

Two measurements per cell:

- **selectivity** — twin-pair accuracy on real concepts minus twin-pair accuracy
  on random directions, at full coverage;
- **abstention** — the same gap as the least-confident cells are dropped, the
  measurement `29` and `31` turn on.

## What each outcome would mean

**All three boosts collapse selectivity.** Then it is a law of this setup, not a
property of any one method: **you cannot make a model introspect better without
making it worse at telling meaning from noise**, and the field's standard
false-positive control cannot see the cost. Strongest and most useful outcome.

**Ablation and prompting preserve it, training does not.** Then the collapse is
specific to training — the only boost that changes weights — and there is a safe
way to elicit introspection. That is a positive result and this project has none.

**Prompting collapses it but ablation does not, or vice versa.** The interesting
middle. Whatever distinguishes them is the mechanism, and both are inference-time
so the difference cannot be about weights.

**Nothing moves at all.** The boosts do not transfer to a 3B model on this task,
and the run says something about scale rather than about introspection. `32`'s
gate makes this diagnosable rather than ambiguous for the ablation arm at least.

## Kill rule

If the `none` arm does not reproduce `14`'s published content accuracy within
±0.10, the apparatus has drifted and nothing else in the run is comparable. Stop
and fix before reading any boost arm.

## Prediction, on the record

**Prompting preserves selectivity; ablation preserves it; training does not.**
About 55/45, and I hold it weakly.

The reasoning from `31`: training collapses selectivity because it teaches a new
capability — reading the *direction* of any displacement cleanly — and that skill
is indifferent to meaning. Prompting and ablation both act on an existing
capability rather than installing one, so they should scale what is there,
including the existing preference for real concepts.

The reason I hold it weakly is that `24` already showed a prompt that measurably
improves engagement with this task bought exactly nothing on the thing that
mattered. Instructions land without changing what is available.

## Cost

Three conditions × two arms × four concept pairs × 24 episodes = 576 episodes,
plus the direction build and its gate. Inference only, no training, one model
load. About fifteen minutes.

---

# Result: a clean pattern, and not enough data to believe it yet

Run **2026-08-13**, 576 episodes. Artifacts: `results/three_boosts_v1_raw.jsonl`,
`results/three_boosts_v1_summary.json`.

**Anchor passes.** The `none` arm gives content accuracy 0.896 against `14`'s
published 0.899. Format rate 1.000 in every cell.

## The point estimates

Twin-pair accuracy, 48 pairs per cell:

| condition | content | random | **gap** | mean margin |
|---|---:|---:|---:|---:|
| `none` | 0.792 | 0.312 | **+0.479** | 4.51 |
| `prompt` | 0.896 | 0.417 | **+0.479** | 7.64 |
| `ablate` | 0.604 | 0.188 | +0.417 | 2.93 |

**Prompting lifts both arms by exactly the same amount** — content +0.104, random
+0.104 — and the selectivity gap is unchanged to three decimals. Confidence rises
substantially (margin 4.51 → 7.64), so the prompt is doing something real to the
model's certainty.

Set against training, from [`08`](08-sensitivity-specificity-tradeoff.md) and
[`31`](31-why-training-inverts-abstention.md), the pattern is a clean three-way
dissociation:

| boost | selectivity gap | abstention |
|---|---|---|
| `prompt` (inference-time) | **unchanged**, 0.479 → 0.479 | works: gap widens to +0.600 |
| `ablate` (inference-time) | roughly unchanged, and not a boost | works: gap widens to +0.800 |
| **training** (weights) | **collapses**, 0.232 → 0.045 | **inverts**: gap narrows to 0.019 |

Training is the odd one out on **both** axes, and it is the only boost that
changes weights. That is exactly what `31`'s mechanism predicts: training installs
a new capability — reading any displacement's direction cleanly — which is
indifferent to meaning, while an inference-time intervention scales what is
already there and keeps the meaning-sensitivity with it.

## And now the part that matters

**None of the individual comparisons is significant.** Bootstrap, 48 twin pairs
per cell:

| comparison | estimate | 95% CI |
|---|---:|---|
| prompt − none, content | +0.104 | **[−0.042, +0.250]** |
| prompt − none, random | +0.104 | **[−0.083, +0.292]** |
| **change in selectivity gap** | **+0.000** | **[−0.229, +0.250]** |

The gap change is exactly zero as a point estimate, and its interval is wide
enough to contain training's −0.187. **So this run cannot distinguish "prompting
preserves selectivity" from "prompting collapses it exactly as much as training
does."** The dissociation above is a pattern in point estimates, not an
established finding, and describing it as one would be the error this repository
keeps catching.

I am recording it as **suggestive and underpowered**, and the numbers that make it
suggestive — two arms moving by the identical +0.104, a gap change of 0.000 — are
the kind of tidiness that regresses.

## My prediction, scored

I predicted prompting and ablation preserve selectivity while training does not,
at 55/45 and held weakly. **The point estimates match exactly.** But I claimed a
weak prior and the data are too weak to promote it, so this scores as *consistent
with*, not *confirmed*.

## What fixes it, and it is cheap

The design uses one carrier. `CONFIRM_VISIBLE_SAMPLES` has **three**. Running all
three triples n to 144 twin pairs per cell and shrinks every interval above by
about 42% — enough to separate a preserved gap from training's collapse.

That is [`34`](34-three-boosts-powered.md), and it is the obvious next run rather
than a new idea. **A pattern this clean deserves either confirmation or death, and
at 48 pairs it gets neither.**

## Limits

One model, one layer, four concept pairs, one carrier. The training comparison is
against `08` and `31`'s separate runs rather than a head-to-head arm, because the
adapters were never saved. The abstention curves are computed on 48 pairs and are
correspondingly noisy at low coverage — the `ablate` arm's +0.800 at 20% coverage
rests on about 10 cells.
