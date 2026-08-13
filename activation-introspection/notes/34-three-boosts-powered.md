# 34 — The same run, with enough data to believe it (pre-run note)

Written before the run. This is not a new idea; it is
[`33`](33-three-boosts-one-control.md) at three times the sample size, because
`33` produced a pattern too clean to ignore and too weak to report.

## Why this exists

`33` found that prompting lifts introspection on real concepts and on
magnitude-matched random directions by **exactly the same amount** — +0.104 each —
leaving the selectivity gap unchanged at +0.479. Against training, which collapses
that gap from 0.232 to 0.045, that is a three-way dissociation with a mechanism
behind it from [`31`](31-why-training-inverts-abstention.md).

And none of it is significant. At 48 twin pairs per cell:

| comparison | estimate | 95% CI |
|---|---:|---|
| prompt − none, content | +0.104 | [−0.042, +0.250] |
| change in selectivity gap | +0.000 | **[−0.229, +0.250]** |

That last interval contains training's −0.187. **The run cannot tell "prompting
preserves selectivity" apart from "prompting destroys it exactly as training
does",** which is the entire question.

## What changes

**One thing: the number of carriers, from one to three.**
`CONFIRM_VISIBLE_SAMPLES` has three; `33` used the first. Using all three triples
n to **144 twin pairs per cell** and shrinks every interval by about 42%.

Nothing else moves. Same conditions, same arms, same concept pairs, same episode
machinery, same scoring, same anchor check against `14`'s published 0.899, same
kill rule. The carrier is part of the twin key already, so cells from different
carriers cannot collide.

## The pre-registered test

**Does the change in selectivity gap between `prompt` and `none` have a 95%
interval excluding training's −0.187?**

- **Yes, and the estimate is near zero** → prompting preserves selectivity where
  training destroys it. The dissociation is real: **the cost is specific to the
  boost that changes weights.** That is the result the whole 29→31→32→33 line has
  been reaching for.
- **Yes, and the estimate is near −0.187** → prompting collapses selectivity too,
  and the cost belongs to boosting in general rather than to training. A stronger
  and more alarming claim, and the more useful one for safety.
- **No — the interval still spans both** → 144 pairs is still not enough, and the
  honest move is to stop rather than run a third time chasing significance. Say
  that this design cannot answer the question at the scale this machine allows.

That third branch is a real possibility and it is written down now so that a
null does not get quietly reframed as support for whichever side the point
estimate lands on.

## Prediction, on the record

**The interval will exclude −0.187 and the estimate will stay near zero**, about
65/35. `33`'s two arms moved by identical amounts, which is unlikely to be pure
coincidence, and the mechanism from `31` predicts it independently.

I also expect the point estimates to move somewhat — 48 pairs is small enough that
+0.104 and +0.104 landing identically is partly luck, and I would not bet on that
tidiness surviving.

## What a boring result buys

If the intervals still span both hypotheses, that is worth knowing before anyone
builds on `33`'s table. The failure mode this guards against is quoting a clean
three-way dissociation from point estimates that cannot support it — which is
exactly what [`13`](13-shared-axis-audit.md), [`15`](15-matched-reader-on-content.md)
and [`26`](26-someone-elses-rules.md) each did in their own way.

## Cost

1,728 episodes, three carriers, plus the direction build. Inference only, one
model load, about forty minutes.

---

# Result: `33` was one carrier's accident, and the reason is worth more than the
# result

Run **2026-08-13**, 1,728 episodes, 144 twin pairs per cell. Artifacts:
`results/three_boosts_v2_raw.jsonl`, `results/three_boosts_v2_summary.json`.

**Anchor is exact**: `none` content accuracy **0.899** against `14`'s published
**0.899**. Format 1.000 everywhere.

| condition | content | random | gap |
|---|---:|---:|---:|
| `none` | 0.799 | 0.111 | **+0.688** |
| `prompt` | **0.618** | 0.139 | **+0.479** |
| `ablate` | 0.722 | 0.132 | +0.590 |

`33` reported prompting *raising* content from 0.792 to 0.896. With three carriers
instead of one it *lowers* it, 0.799 to 0.618. Change in selectivity gap:
**−0.208, 95% CI [−0.340, −0.083]** — excludes zero, and comfortably contains
training's −0.187.

**So `33`'s headline reverses.** Prompting does not preserve selectivity here.

## But that is not the right description either

Broken out by carrier — the variable the design treats as nuisance:

| carrier text | `none` content | `prompt` content | gap `none` | gap `prompt` |
|---|---:|---:|---:|---:|
| "A routine maintenance record is open…" | 0.792 | 0.896 | +0.688 | +0.646 |
| "After lunch, the clerk placed an unsigned form…" | 0.604 | 0.833 | +0.479 | +0.667 |
| **"Nothing changed. Hidden state marker: §"** | **1.000** | **0.125** | **+0.896** | **+0.125** |

On two carriers the prompt helps and the gap survives. On the third it converts a
**perfect** cell into a below-chance one. The pooled −0.208 is that single carrier.

`33` used the first carrier. That is why it saw the opposite.

## The diagnosis, and it is clean

The `introspect` wording opens: *"Something has been added to your internal state
at each observation below."* The third carrier's visible text reads: **"Nothing
changed."**

The model resolves the contradiction by giving up on the state:

| | constant-label cells | mean confidence | twin-pair |
|---|---:|---:|---:|
| `none` | **0/48** | 7.43 | 1.000 |
| `prompt` | **42/48** | 10.72 | 0.125 |

Zero constant-label cells becomes 42 of 48, and **confidence goes up** while
accuracy collapses. It is not confused; it is confidently answering a different
question.

## What this actually establishes, which is a methodological finding

Every design in this repository leans on one structural control: the visible text
is byte-identical across twins, so **an input-only learner is pinned at exactly
0.500 by construction**. That control is real and it holds here.

**It is also narrower than it looks.** It guarantees the text cannot *predict the
label*. It does nothing to stop the text from *disabling the mechanism* — and here
a semantic clash between the instruction and supposedly-inert filler took a cell
from 1.000 to 0.125 without touching the injected states at all.

That matters beyond this repository. Prompting boosts are reported as averages
across inputs. This is a case where one prompt inverts performance on a subset of
inputs for a plainly semantic reason, and pooling hides it entirely. Anyone
reporting "prompt X improves introspection by N%" without a per-input breakdown
could be averaging over exactly this.

## The pre-registered test, and an admission about it

The rule was: *does the interval exclude training's −0.187?* It does not —
[−0.340, −0.083] contains it — which routes to the branch I wrote as "stop rather
than run a third time chasing significance."

**I am stopping, but the pre-registration was badly drafted and I am saying so.**
Branch 2 read "yes, and the estimate is near −0.187", which is self-contradictory:
an interval cannot exclude a value the estimate sits on. The test I should have
written is whether the interval excludes **zero**, and by that test the answer is
clear — prompting does not preserve the gap.

That drafting error did not change the decision, because the carrier breakdown
answers the substantive question either way. But a malformed pre-registration is
exactly the sort of thing that lets someone pick the convenient reading
afterwards, and it is recorded rather than tidied.

## Predictions, scored

I predicted 65/35 that the interval would exclude −0.187 with the estimate near
zero. **Wrong on both counts**: the estimate is −0.208 and the interval contains
−0.187.

I also predicted the point estimates would move because `33`'s tidiness was partly
luck. **Right, and by more than I meant** — I expected drift, not a sign flip.

## What this closes

**The 29→31→32→33 line stops here**, and not for lack of power. It stops because
the effect it was chasing is not stable across a variable the design treats as
irrelevant, and at this scale that instability is larger than the effect. Three
carriers is enough to show the instability; it is not enough to characterise it,
and characterising it would need a carrier bank built for that purpose.

What survives, and it is the durable part:

- [`29`](29-can-abstention-recover-selectivity.md) and
  [`31`](31-why-training-inverts-abstention.md) stand — they are measured on
  training artifacts across seeds, not on this carrier-sensitive design.
- **The semantic-clash finding above**, which is new, diagnosed, and a caution
  about how prompting results are reported.
- `32`'s boundary: the refusal-ablation boost does not transfer to this scale.

## Limits

One model, three carriers, four concept pairs. Three carriers is a very small
sample of a variable that turns out to matter enormously, so "two of three help"
is not a rate. The semantic clash is diagnosed from one carrier and one prompt
family; that it is *the* mechanism is inferred from the constant-label jump and
the text itself, and would need a deliberate clash-versus-no-clash design to be
established.
