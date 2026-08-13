# 36 — Is it the stance, or is it the sentence? (pre-run note)

> **Bounded 2026-08-13 by [`37`](37-is-it-the-relation-or-the-instruction.md).**
> The carrier side of this result is general — nine strings, three stances. **The
> instruction side is not.** Two further instructions that also assert internal
> change behave differently: `injected` shows a carrier effect of only −0.156
> against `introspect`'s −0.729, and `feels` degrades on *every* carrier and shows
> no carrier sensitivity at all. So this is largely a property of `24`'s
> `introspect` wording interacting with carrier stance, **not** of
> instruction–context contradiction in general. Read every "instruction that
> asserts internal change" below as "this instruction".

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

---

# Result: it is the stance. Nine strings, perfect separation.

Run **2026-08-13**, 1,728 episodes, 144 twin cells per stance × instruction.
Artifacts: `results/prompt_clash_v2_raw.jsonl`,
`results/prompt_clash_v2_summary.json`.

**Anchor holds**: `35`'s original string reproduces its own collapse.

Twin-pair accuracy, per string, `silent` → `asserts`:

| carrier string | silent | asserts | drop |
|---|---:|---:|---:|
| **`denies`** "Nothing changed." | 1.000 | 0.125 | **−0.875** |
| **`denies`** "The state is exactly as it was before." | 0.875 | 0.229 | **−0.646** |
| **`denies`** "No alteration has been made here." | 0.812 | 0.167 | **−0.646** |
| `neutral` "A routine maintenance record is open." | 0.792 | 0.896 | +0.104 |
| `neutral` "The afternoon shift log sits on the desk." | 0.771 | 0.917 | +0.146 |
| `neutral` "A printed inventory list lies nearby." | 0.896 | 0.708 | −0.188 |
| `affirms` "Something is different now." | 0.938 | 0.854 | −0.083 |
| `affirms` "The state has been altered from before." | 0.792 | 0.771 | −0.021 |
| `affirms` "An adjustment has been made here." | 0.896 | 0.896 | 0.000 |

**All three denials collapse below the 0.25 coin-flip null. None of the six others
comes close.** Perfect separation across nine strings.

Pooled, 144 twin cells per cell:

| stance | effect of the instruction | 95% CI |
|---|---:|---|
| `denies` | **−0.722** | **[−0.799, −0.639]** |
| `neutral` | +0.021 | [−0.069, +0.111] |
| `affirms` | −0.035 | [−0.118, +0.049] |

**Interaction −0.715, 95% CI [−0.812, −0.611].**

## What this establishes

**The effect belongs to the stance, not to the sentence.** That was the one hole
`35` named in itself, and it is now closed with three strings per stance at matched
length and register. Neutral and affirming carriers are untouched by the same
instruction — both intervals comfortably contain zero.

So the claim stands at its full strength:

> An instruction asserting that something was added to the model's internal state
> **destroys the readout on any carrier text that denies change**, converting a
> near-perfect forced choice into a confident constant response. It leaves
> carriers that are neutral or that agree with it alone.

## My prediction, scored

I predicted all three would collapse but **not equally**, at 60/40, with the stark
original collapsing hardest and discursive denials milder — and said that if so,
"directness of contradiction" would be the finer-grained claim.

**The count is right and the gradient is barely there.** The original drops 0.875
and the two discursive denials drop 0.646 each — identical to one another, and all
three land below chance. There is a hint that the starkest string is worst, on a
difference of 0.23 with intervals I have not computed per-string and would not
trust at 48 cells each.

**So the directness sub-claim is not supported.** Denying change discursively is
about as destructive as denying it in three words, and I should not have expected
otherwise: the model is not weighing rhetorical force, it is resolving a
contradiction.

The pooled interaction also shrank from `35`'s −0.885 to −0.715, exactly as adding
two less-extreme members of a class should. `35`'s number was the best case, not
the typical one.

## Where this leaves the repository

This is now **the most solidly established result here**: nine strings, a
pre-registered per-string test, matched length, a reproduced anchor, and an
interaction whose interval is nowhere near zero.

It is still, as [`35`](35-when-the-prompt-contradicts-the-page.md) records, an
**instance of a published phenomenon** — instruction–context conflict degrading
behaviour is prior art. What is ours is the application and the caution: an
introspection elicitation prompt is subject to it, the failure mode is a
*confident* collapse to one label rather than visible degradation, and the pooled
averages this literature reports its prompting gains as would hide it completely.

## Limits

One model, one instruction family, three concept pairs' worth of carriers per
stance, one injection layer. The instruction family is `24`'s `introspect`
wording; whether other introspection prompts that assert internal change do the
same is untested, and that is now the sharpest remaining hole rather than the
carrier one. Nothing here says the effect exists at the scales where the published
prompting gains were measured.
