# 36 — Is it the stance, or is it the sentence? (pre-run note)

Written before the run.

## The weakness this attacks

[`35`](35-when-the-prompt-contradicts-the-page.md) is the cleanest measured result
in this repository: an instruction asserting that something was added to the
model's internal state takes a cell from **48/48 to 6/48** when the carrier text
reads "Nothing changed", and does nothing of the kind on carriers that are neutral
or that affirm change. Interaction −0.885, CI [−1.010, −0.750].

It also has one obvious hole, which `35` states in its own limits:

> One carrier per stance, so **"carriers that deny change" is a class of one**, and
> the effect could belong to that sentence rather than to its stance.

Everything in this repository that rested on a class of one has died —
[`26`](26-someone-elses-rules.md) on rules I wrote, [`33`](33-three-boosts-one-control.md)
on a single carrier, [`30`](30-does-it-know-it-is-about-to-be-wrong.md)'s hard-band
lead. **A result this clean, resting on one sentence, is exactly the shape of the
things that have died.** So it gets the same test before it gets believed.

## What I am about to do

`35` unchanged, except **three carrier strings per stance instead of one**.

| stance | strings |
|---|---|
| `denies` | `35`'s original, plus two more that deny change in different words |
| `neutral` | `35`'s original, plus two more that assert nothing either way |
| `affirms` | `35`'s original, plus two more that assert change |

Written to the same constraints as `35`: comparable length and register across
stances, so length stays controlled, and each ends with the same
`Hidden state marker: §` so the injection site is identical.

The instruction arms are unchanged — `silent` and `asserts`, verbatim `24`'s
`baseline` and `introspect` families. Same concept pairs, same strength, same
twin-pair scoring, same constant-label diagnostic.

## The pre-registered test

**Does the collapse appear on all three `denies` strings, and on none of the six
others?**

Reported as the interaction from `35`, recomputed with stance as a three-level
factor over three strings each — and, more importantly, as the **per-string
table**, because the whole point is whether the effect is a property of the class
or of one member.

- **All three `denies` strings collapse, the six others do not.** The effect is the
  stance. `35` stands and is no longer a class of one.
- **Only `35`'s original string collapses.** The effect belongs to that sentence.
  `35`'s interpretation is withdrawn and what remains is a curiosity about one
  string — which is worth knowing, and much less.
- **Some `denies` collapse and some do not.** Partial: something narrower than
  "denies change" is doing the work, and the split between the strings is the clue
  to what. Report as partial and do not round it up.
- **Collapse appears on non-`denies` strings too.** The stance account is wrong and
  the real variable is something these strings share that I have not identified.

## Prediction, on the record

**All three `denies` strings collapse, though not all as completely as the
original.** About 60/40.

Lower than `35`'s 70/30 despite `35` having confirmed, because the original
string is unusually stark — "Nothing changed." is a *direct* denial of the
instruction's claim, three words long, with no other content to attend to. A
carrier that denies change more discursively gives the model somewhere else to
put its attention, and the collapse may be milder or absent.

If that is what happens — original collapses hard, the other two mildly — the
honest reading is that **directness of the contradiction matters**, which is a
finer-grained claim than `35` makes and would need its own test.

## Kill rule

If `35`'s original string does not reproduce its own collapse (constant-label rate
above 0.5, twin-pair below 0.3), the effect is not stable across runs at all and
nothing in `35` or here is interpretable. That cell is the anchor.

## Cost

3 stances × 3 strings × 2 instructions × 4 concept pairs × 24 episodes = 1,728
episodes. Inference only, one model load, about half an hour.

## What would change my mind about running it at all

Nothing — this is the cheapest possible test of the single largest weakness in the
one clean result here, and every outcome above changes what may be said.
