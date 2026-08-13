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
