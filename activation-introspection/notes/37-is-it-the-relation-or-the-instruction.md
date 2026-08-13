# 37 — Is it the relation, or is it that one instruction? (pre-run note)

Written before the run.

## The hole this attacks

[`36`](36-is-it-the-stance-or-the-sentence.md) closed the carrier side properly:
three strings per stance, matched length, perfect separation, interaction −0.715
with a 95% interval of [−0.812, −0.611]. The effect is the carrier's **stance**,
not any one sentence.

It closed that hole and named the next one in its own limits:

> One instruction family… whether other introspection prompts that assert internal
> change do the same is untested, and that is now the sharpest remaining hole.

Symmetry demands it. `36` proved the effect is not about one carrier sentence by
varying carrier sentences. **The identical argument applies to the instruction**,
and right now "instructions that assert internal change" is a class of one — the
exact situation `36` was written to fix on the other side.

## What I am about to do

Cross **instruction stance** with **carrier stance**, three strings on each side
where it matters.

| instruction | asserts internal change? | source |
|---|---|---|
| `introspect` | **yes** | `24`'s family, the one `35`/`36` used |
| `injected` | **yes** | new: says an edit was made to its activations |
| `feels` | **yes** | new: says its internal state will feel different |
| `baseline` | no | `24`'s family, infer the mapping, no claim |
| `eliminate` | no | `24`'s family, two exhaustive labels, no claim |

Carriers: **`denies`** and **`neutral`**, two strings each, taken verbatim from
`36`. `affirms` is dropped — `36` showed it behaves like `neutral`, and the
budget is better spent on the instruction side, which is the untested one.

New instruction families are defined **locally in this runner**, not added to
`run_heldout_elicitation`'s `FAMILIES`, so `24`'s published artifact is untouched.
As always, only the header lines differ; every line carrying an injection site is
byte-identical, so an input-only strategy stays pinned at 0.500 by construction.

## The pre-registered test

**Do all three asserting instructions collapse on `denies` and neither
non-asserting one, with all five leaving `neutral` alone?**

The prediction is a 2 × 2 pattern with one full cell affected: asserting ×
denying collapses, the other three cells do not.

- **All three assert-instructions collapse on `denies`, neither other does, and
  `neutral` is untouched throughout.** The effect is the *semantic relation between
  instruction and context*, established on both sides. That is a general finding
  and the claim can be stated without hedging about particular strings.
- **Only `introspect` collapses.** The effect belongs to that instruction, `35` and
  `36` describe a property of one prompt, and the "relation" framing is withdrawn.
- **A non-asserting instruction collapses too.** The account is wrong — something
  other than contradiction is driving it, and whatever `baseline` or `eliminate`
  shares with the asserting prompts is the real variable.
- **Everything collapses on `denies`.** That carrier stance is simply fragile under
  any instruction change, which would make `36`'s clean separation across
  instructions-of-one a fluke worth understanding.

## Prediction, on the record

**All three asserting instructions collapse on `denies`; the two non-asserting
ones do not; `neutral` is untouched.** About 70/30.

Higher than `36`'s 60/40 because `36` came in clean and because the mechanism —
the model resolving a contradiction between what it is told about itself and what
the page says — does not obviously depend on which words carry the assertion.

The 30% is that `introspect` does more than assert change. It also says *"attend to
how each one feels from the inside"*, which is an instruction to introspect as well
as an assertion about state. If the collapse needs both, the two new families —
which assert without that directive — will be milder or clean, and the honest
reading becomes that **asserting change is not sufficient; you must also redirect
attention inward.** That would be a finer and more interesting claim than the
current one.

## Kill rule

If `introspect` × `denies` does not reproduce `36`'s collapse, the effect is not
stable across runs and nothing here or in `35`/`36` is interpretable. That cell is
the anchor and it is checked first.

## Cost

5 instructions × 2 carrier stances × 2 strings × 4 concept pairs × 24 episodes =
1,920 episodes. Inference only, one model load, about thirty-five minutes.
