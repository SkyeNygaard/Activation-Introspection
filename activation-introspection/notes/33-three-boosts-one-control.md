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
