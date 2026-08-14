# 38 — Does introspection training report concept identity, or the fact of displacement? (pre-run note)

Queued **2026-08-14**. Not run.

**This is not a notes/29–37 descendant.** The handoff bans another prompt-conflict
variant and it is right to. This changes the intervention, not the wording: it is
the first note here that acts on *training* rather than measuring its output.

## The question

Every study in this repository measures what a trained reporter *says*. None asks
what it *uses*. Those come apart in a specific way that matters:

- **Identity.** The adapter reads which concept the injected direction encodes.
- **Displacement.** The adapter reads that the residual stream was pushed off its
  usual path, and the concept label rides along on whatever the demonstrations
  happened to pair with that.

Both produce high accuracy on every task run here so far. The 2026-08-14
correction pass makes this urgent rather than academic: the claim that training
destroys selectivity was withdrawn precisely because the arm meant to separate
these two possibilities turned out not to separate them.

## What I am about to do

Take the direction that encodes *an injection happened at all*, and remove it —
first at evaluation, then during training — and see whether the reports survive.

1. Capture residual states under matched conditions: clean, and injected with
   `target`, `random` and `shuffled` directions, same carriers, same positions.
2. Compute the displacement direction: mean(injected) − mean(clean), pooled over
   concepts so that what survives is what all injections share.
3. **Gate.** Check that direction actually separates injected from clean. Report
   a held-out separation score.
4. If it does: project it out of the residual stream and re-score the existing
   trained reporters. Then train one adapter with it projected out throughout.

## Why this is worth the time

It is the only question here whose answer changes what someone else should build.
[Introspection Adapters](https://arxiv.org/pdf/2604.16812) names a high false
positive rate as its first limitation and reducing it as future work. If reports
survive removing displacement, the adapter reads identity and the false-positive
worry is about something else. If they collapse, an introspection adapter is a
displacement detector wearing a concept vocabulary, and the fix is architectural
rather than a matter of more training data.

It also borrows a method with a track record on a different problem —
[Casademunt et al.](https://arxiv.org/abs/2507.16795) steered how a model
generalises after fine-tuning by ablating concepts, with no change to the data or
the loss — and points it at a question nobody has pointed it at.

## What each outcome means, including the boring one

| result | reading |
|---|---|
| Reports survive ablation, identity accuracy roughly intact | The adapter reads identity. Strongest outcome, and it says the false-positive problem is not displacement-driven |
| Reports collapse to the constant-label floor | The adapter was riding on displacement. A mechanism claim, and a warning about building on adapters |
| Ablated training keeps identity **and** reduces confident answers when nothing is injected | Best case: an improvement to introspection adapters on the axis their authors named, obtained without touching data or loss |
| No change either way | The pooled direction is not what the adapter uses. Kills a live hypothesis; report it |

**Kill rule, declared before the run.** If step 3 shows the pooled displacement
direction does not separate injected from clean on held-out rows, stop. There is
no coherent thing to ablate and the rest of the design is void. Report the null
and do not repair it by picking a different direction after seeing the data.

## What it costs

- Capture and direction fit: minutes of inference. **Pilot at 0.5B first** — it is
  the only size that fits current free memory — then 3B.
- Re-scoring existing reporters: inference only.
- One adapter trained with ablation: ~47 min at 3B, and needs the machine cleared.

## Declared in advance

- The pooled direction is fitted on development rows and evaluated on held-out
  concepts and carriers. The split is declared here, before any of it is computed.
- Ablation is applied at the injection site, all state positions, matching how the
  edits themselves are applied.
- The 0.5B pilot is a plumbing and capacity check, not a result. If 0.5B cannot do
  the anchor task, that says nothing about 3B and will not be reported as if it did.

## Pilot, 0.5B, 2026-08-14 — the gate passes and the bound is tight

`results/displacement_direction_pilot_qwen05b_v2.json`. Qwen2.5-0.5B, inject at
layer 6 of 24, read the final block, strength 1.0, 8 concepts × 3 arms × 3
carriers per split. Fitted on development concepts *and* development carriers,
scored on held-out both.

| | held-out |
|---|---:|
| separation of injected from clean (AUROC) | **1.000** |
| share of displacement energy along the mean delta | **0.217** |
| share along the leading component | **0.217** |

Two things follow, and the second matters more.

**The direction is real.** It orders every held-out injected state above every
held-out clean one. And because the two shares are equal to three decimals, the
mean delta *is* the leading component — the shared "an injection happened" offset
is the dominant axis, not an artifact of concept-specific structure sitting on
top of it.

**But it is only a fifth of the effect.** Removing that one direction leaves
about 78% of what an injection does to the final state untouched. So the planned
rank-1 ablation cannot support the reading the design wanted: "the reports
survived" would be unsurprising when most of the displacement is still there.

This is the weakness declared below, now with a number on it. Two honest ways on:

1. **Ablate a subspace, not a direction.** Take the leading components of the
   injected-minus-clean deltas until a declared fraction of the energy is gone,
   then ablate all of them. `Intervention` currently ablates rank-1 only, so this
   needs a small extension — and the fraction must be fixed before the run, not
   tuned until the result is interesting.
2. **Keep rank-1 and report the bound alongside every number.** Cheaper, weaker.

I prefer (1). The rank needed to reach a declared fraction is itself a result —
"the fact of an injection occupies k dimensions at the readout" is a cleaner
statement than anything the rank-1 version could produce.

**Not yet checked at 3B.** This is a 0.5B pilot at one layer and one strength; it
is a plumbing and design check, not a finding, and the 0.217 may not transfer. The
3B run is the one that decides the rank.

## Known weakness, stated now rather than found later

"An injection happened" may not be one direction. If it is genuinely
multi-dimensional, a rank-1 projection removes part of it and a partial collapse
becomes uninterpretable — it could mean the adapter half-uses displacement, or
that the ablation half-worked. The gate in step 3 is the check: alongside the
separation score, record how much of the injected-versus-clean variance the first
component explains. If that fraction is low, say so and treat every downstream
number as bounded by it.
