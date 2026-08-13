# 35 — When the prompt contradicts the page (pre-run note)

Written before the run.

## The observation this is built to test

[`34`](34-three-boosts-powered.md) found that one instruction, applied to three
carrier texts that the design treats as interchangeable filler, produced this:

| carrier text | without the prompt | with it |
|---|---:|---:|
| "A routine maintenance record is open…" | 0.792 | 0.896 |
| "After lunch, the clerk placed an unsigned form…" | 0.604 | 0.833 |
| **"Nothing changed. Hidden state marker: §"** | **1.000** | **0.125** |

On the third, constant-label cells went from **0 of 48 to 42 of 48**, twin-pair
accuracy from 1.000 to 0.125, and mean confidence *up* from 7.43 to 10.72.

The instruction opens *"Something has been added to your internal state at each
observation below."* The carrier says **"Nothing changed."**

**The obvious reading is a semantic clash**, and `34` labels it inferred rather
than measured, because it rests on one carrier and one prompt. This measures it.

## Why it is worth measuring rather than asserting

If it is real it is a caution with reach beyond this repository. Prompting boosts
in this literature are reported as averages over inputs — 0.3% → 39.9%, 10.8% →
63.8%. **A prompt that inverts performance on a semantically-clashing subset would
be invisible in every one of those numbers.** Nobody reports a per-input
breakdown, and until `34` I had no reason to think one was needed.

If it is not real — if the third carrier is odd for some other reason — then `34`'s
most interesting paragraph is wrong and should be struck.

## What I am about to do

A deliberate 3 × 2. **The carrier text's *claim about change* is crossed with the
instruction's *claim about change*.** Nothing else moves: same concept pairs, same
injection sites, same strength, same twin-pair scoring, same episode machinery.

| carrier stance | text |
|---|---|
| `denies` | "Nothing changed. Hidden state marker: §" |
| `neutral` | "A routine maintenance record is open. Hidden state marker: §" |
| `affirms` | "Something is different now. Hidden state marker: §" |

| instruction stance | header |
|---|---|
| `silent` | `24`'s `baseline`: infer the mapping, no claim about internal state |
| `asserts` | `24`'s `introspect`: "Something has been added to your internal state" |

`denies` and `neutral` are the exact strings `34` used. `affirms` is new, and it is
the cell that makes this a test rather than a demonstration: if the mechanism is
*contradiction*, then an instruction that asserts change should be **fine or better**
on a carrier that also asserts change, and bad only where the carrier denies it.

**Primary measure: the constant-label rate**, because that is what actually moved
in `34` — 0 to 42 of 48. Twin-pair accuracy is reported alongside. A collapse to
constant labelling is a specific, diagnosable failure, not a general accuracy dip.

## What each outcome would mean

**Collapse confined to `denies` × `asserts`.** The mechanism is contradiction, it
is measured rather than inferred, and there is a concrete warning for anyone
reporting a pooled prompting gain: **your prompt may be silently destroying the
readout on inputs that contradict it.**

**`denies` is bad under both instructions.** Then the carrier is simply a hard one
and `34`'s clash story is wrong. The 1.000 without the prompt argues against this
outright — it was the *easiest* carrier — but it is the first alternative to rule
out.

**All three carriers collapse under `asserts`.** Then the instruction is just
harmful here at three carriers' worth of evidence, `34`'s per-carrier split was
noise, and the finding is about that prompt rather than about clash.

**Nothing collapses anywhere.** `34`'s cell does not reproduce. That would be the
most alarming outcome about this apparatus rather than about the model, and it
would need chasing before anything else in `33`–`34` is trusted.

## Kill rule

If the `denies` × `asserts` cell does not reproduce `34`'s collapse — constant-label
rate above 0.5 and twin-pair below 0.3 — then the effect is not stable even within
its own cell, and nothing else in this note is interpretable. Report that.

## Prediction, on the record

**Collapse confined to `denies` × `asserts`.** About 70/30.

Specifically: `denies` × `silent` stays near 1.000 with near-zero constant
labelling, reproducing `34`. `affirms` × `asserts` looks like `neutral` × `asserts`
or better. The clash cell collapses.

The 30% is that "Nothing changed" is a *short* carrier as well as a contradicting
one, and shortness is confounded with stance in `34`'s set. **`affirms` is written
to be short too**, so this confound is at least partly controlled by construction —
if collapse tracks stance rather than length, `affirms` will be fine.

## Cost

3 carriers × 2 instructions × 4 concept pairs × 24 episodes = 576 episodes.
Inference only, no training, one model load. About ten minutes.

## What would change my mind about running it at all

If `affirms` cannot be written at a length and register comparable to `denies`,
stance and length stay confounded and the design cannot separate them. Checked
when the strings are written, not after.
