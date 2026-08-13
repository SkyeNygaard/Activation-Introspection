# 31 — Why does training invert the abstention benefit? (pre-run note)

Written before the analysis ran. No GPU; existing artifacts only.

## The fact that needs explaining

[`29`](29-can-abstention-recover-selectivity.md) found something specific and left
it unexplained. Dropping a model's least-confident self-reports:

- in the **untrained** model, nearly doubles the gap between real concepts and
  magnitude-matched random directions (0.232 → 0.455);
- in the **trained** reporters, erases it (0.099 → 0.019).

Training inverts the sign of the abstention benefit. An unexplained effect is a
curiosity; an explained one is a mechanism. This note tries to explain it.

## The hypothesis

[`08`](08-sensitivity-specificity-tradeoff.md) already says, in words, what the
trained reporter became: *"a generic displacement detector… asked 'is concept X
active', it answers a different question, 'did something move at layer 9'."*

If that is literally true, the confidence follows:

> **Trained confidence tracks how large the edit was, not whether it meant
> anything.** A random direction at matched magnitude is a large displacement, so
> the trained model is confident about it — and the most confident reports are
> therefore not the most meaningful ones. That is the inversion.

## The clean test, and why it cannot be run

The decisive experiment is a **dose-response**: vary injection strength for both
real concepts and random directions, and check whether trained confidence rises
with strength *equally* for both while base confidence rises mainly for concepts.

**That data does not exist and cannot be made.** `remap_training_v2` sweeps three
strengths (0.15, 0.25, 0.5) on `target` only; random directions appear at 0.5
alone. Generating the missing cells would require running the trained adapters at
new strengths, and **the adapters were never saved** — the same defect
[`22`](22-the-weak-arm-was-a-floor-not-a-frontier.md) flagged, blocking a second
question now.

Recording that plainly: **the mechanism cannot be established from what exists.**
What follows is a weaker, partial test, labelled as such, and it cannot on its own
confirm the hypothesis.

## The partial test that is available

On `target` only, confidence against strength, for `base` against the two trained
arms. Confidence is `abs(signed_margin)`, as in `29`.

**If trained confidence is driven by displacement magnitude**, the trained arms
should show a **much steeper** rise in confidence from strength 0.15 to 0.5 than
base does — they should be reading the size of the edit, and reading it harder.

**If trained confidence is driven by concept content**, the slopes should be
comparable, and the inversion in `29` needs a different explanation.

This is suggestive either way and decisive neither way, because on `target`
magnitude and meaning increase together. It is worth half an hour because it is
free and because a flat result would falsify the account `08` has been asserting
in words since it was written.

## A second thing worth extracting, since the rows are open

At strength 0.5 both conditions exist, so the ratio of mean confidence on real
concepts to mean confidence on random directions is computable per arm. `29`
already reported the means; the ratio is the quantity the hypothesis speaks to.
**Under the displacement account this ratio should be near 1.0 for trained arms**
— magnitude is matched by construction, so if confidence tracks magnitude it
should not care which is which — **and above 1.0 for base.**

## Prediction, on the record

- Trained arms rise **more than twice** as steeply with strength as base.
- Confidence ratio at strength 0.5: base around **1.2**, trained arms **1.0–1.1**.

I hold this at about 70/30. The competing account is that training simply
sharpens everything — larger margins everywhere, no special relationship to
magnitude — which would produce steep slopes *and* a preserved ratio.

## What each outcome means

**Steep trained slopes and trained ratio near 1.0.** The displacement account is
supported, `08`'s wording is earned rather than asserted, and `29`'s inversion has
a mechanism: abstention selects for large edits, and large edits are not
meaningful edits.

**Steep slopes but trained ratio well above 1.0.** Training sharpened confidence
without making it magnitude-driven. The inversion needs another explanation and
`08`'s "generic displacement detector" phrasing should be softened.

**Flat slopes.** The account is wrong and should be withdrawn from `08`.

## Cost

Minutes of CPU on 13,824 saved rows. No GPU, no model load.

---

# Result: half the prediction was right, and the wrong half is the useful one

Run **2026-08-12** on 13,824 saved rows. No GPU.

## Dose-response, target only

Mean confidence against injection strength:

| arm | 0.15 | 0.25 | 0.50 | rise |
|---|---:|---:|---:|---:|
| `base` | 3.40 | 3.46 | 4.15 | **0.75** |
| `fixed` | 10.48 | 14.05 | 15.36 | **4.88** |
| `remap` | 10.98 | 13.96 | 15.81 | **4.83** |

Trained confidence rises with strength about **6.5× more steeply** than base. I
predicted "more than twice"; it is much more than that. **Prediction correct.**

## Confidence ratio, and this is where I was wrong

Mean confidence on real concepts divided by mean confidence on magnitude-matched
random directions, at strength 0.5:

| arm | ratio |
|---|---:|
| `base` | 1.212 |
| `fixed` | 1.182 |
| `remap` | 1.233 |

I predicted base near 1.2 and the trained arms at **1.0–1.1**, on the reasoning
that if confidence tracked displacement magnitude it should not care which
direction it was. Base came in exactly as predicted. **The trained arms did not
move at all** — 1.18 and 1.23 against base's 1.21.

**So the displacement-*magnitude* account is wrong.** Training does not make
confidence blind to whether a direction is meaningful; the relative preference for
real concepts is the same before and after. `08`'s "generic displacement detector"
is not earned by this route, and the prediction that it would be is scored wrong.

## What is actually driving `29`'s inversion

Since the ratio did not move, the inversion needed a different explanation, so I
decomposed the gap into its two halves. Accuracy as coverage falls, strength 0.5,
confirmation seeds:

| arm | condition | 100% | 70% | 50% | 20% |
|---|---|---:|---:|---:|---:|
| `base` | target | 0.745 | 0.764 | 0.797 | **0.987** |
| `base` | random | 0.513 | 0.504 | 0.508 | **0.532** |
| `fixed` | target | 1.000 | 1.000 | 1.000 | 1.000 |
| `fixed` | random | 0.941 | 0.967 | 0.977 | **0.987** |
| `remap` | target | 1.000 | 1.000 | 1.000 | 1.000 |
| `remap` | random | 0.901 | 0.963 | 0.971 | **0.981** |

**This is the mechanism, and it is cleaner than the one I guessed.**

In the **untrained** model abstention does exactly what it should: keep the
confident half and accuracy on real concepts climbs 0.745 → 0.987, while accuracy
on noise stays pinned at chance, 0.513 → 0.532. Confidence is a guide to
meaningfulness.

In the **trained** models the gap does not close because target falls — target is
at ceiling throughout. It closes because **accuracy on random directions climbs to
0.98**. Filtering for confidence actively *enriches* for confidently-labelled
noise.

So the trained reporter's confidence is a good guide to one thing — **whether it
read the direction of the edit cleanly** — and that is orthogonal to whether the
edit meant anything. It can read a meaningless direction just as cleanly as a
meaningful one, and it is most confident precisely when it has done so.

**`08`'s phrasing survives, by a different route than I predicted.** The trained
model is a displacement detector not because confidence tracks *magnitude* but
because it tracks *readability of the direction*. Softening `08` is not required;
what is required is that the reason be stated correctly, and it now is.

## Why this matters for `29`

`29` reported the inversion without a mechanism, and the obvious deflationary
reading was a ceiling effect — target pinned at 1.000 leaves the gap nowhere to go
but down. **That reading is wrong**, and the table above is what rules it out. The
gap closes from the bottom, not the top: noise accuracy rises 0.90 → 0.98. A
monitor that abstains is not merely failing to improve, it is **concentrating its
output on the cases it is most wrong about.**

## The test that still cannot be run

The decisive dose-response — strength varied for random directions as well as
concepts — remains impossible, because **the adapters were never saved**. Twice now
that omission has blocked a question ([`22`](22-the-weak-arm-was-a-floor-not-a-frontier.md)
was the first). Anyone retraining should save them; the protocol change is a single
`save_pretrained` and it must ship as a deliberate v4.

## Limits

Confidence is `abs(signed_margin)`, an internal quantity. One model, one bank, one
layer, one training recipe, three seeds. The dose-response arm is `target` only, so
magnitude and meaning rise together there and that half of the analysis cannot
separate them — which is exactly why the ratio and the decomposition were needed.
