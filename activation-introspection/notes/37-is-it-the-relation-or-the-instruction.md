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

---

# Result: it is that one instruction. The relation framing is withdrawn.

Run **2026-08-13**, 1,920 episodes, 96 twin cells per instruction × stance.
Artifacts: `results/instruction_stance_v1_raw.jsonl`,
`results/instruction_stance_v1_summary.json`.

**Anchor holds**: `introspect` × `denies` gives 0.177 twin-pair against `36`'s
0.174. The effect is stable across runs.

| instruction | asserts change? | denies | neutral | carrier effect | 95% CI |
|---|---|---:|---:|---:|---|
| **`introspect`** | yes | **0.177** | 0.906 | **−0.729** | **[−0.823, −0.635]** |
| `injected` | yes | 0.698 | 0.854 | −0.156 | [−0.271, −0.042] |
| `feels` | yes | 0.354 | **0.323** | +0.031 | [−0.104, +0.167] |
| `baseline` | no | 0.938 | 0.781 | +0.156 | [+0.062, +0.250] |
| `eliminate` | no | 0.958 | 0.896 | +0.062 | [−0.010, +0.135] |

**The pre-registered 2 × 2 does not hold.** Two things break it:

- **`injected` barely moves.** It asserts internal change in plain terms — "an edit
  has been applied to your activations" — and its carrier effect is −0.156 against
  `introspect`'s −0.729. The difference between them is **−0.573, CI [−0.719,
  −0.417]**.
- **`feels` collapses on *both* carriers**, 0.354 and 0.323. It is not
  carrier-sensitive at all; it is simply a destructive instruction. Its carrier
  effect is +0.031 — nothing.

Against `feels` the gap is **−0.760, CI [−0.927, −0.594]**.

## What is withdrawn

**The framing that `35` and `36` invited — that this is about the semantic relation
between an instruction and its context — is not supported.** Three instructions
that all assert internal change behave in three different ways: one collapses
specifically on denying carriers, one barely responds to carrier stance, and one
degrades everywhere regardless.

So the honest statement narrows to:

> **`24`'s `introspect` wording** loses its readout on carriers that deny change.
> The carrier side of that is general — `36` established it across nine strings —
> but the instruction side is not. It is largely one prompt.

`36`'s result stands exactly as measured. What does not stand is the generalisation
over instructions that I was one step away from making, and that
[`CLAIMS.md`](../../spar-application/CLAIMS.md) had already been written to make
before this ran. That row is being corrected.

## What survives, and it is not nothing

**No non-asserting instruction hurt anywhere.** `baseline` and `eliminate` sit at
0.938/0.781 and 0.958/0.896 across four cells. Whatever is happening needs an
instruction that makes a claim about the model's internal state; it does not happen
by accident.

**And the practical caution is untouched.** Two of three instructions that assert
internal change caused real damage — `introspect` catastrophically on a subset of
inputs, `feels` moderately on all of them. **A prompt written to elicit
introspection can silently cost you more than it buys, and which way it fails is
not predictable from its stated intent.** That is the transferable point, and it
does not require the relation story.

## My prediction, scored

I predicted the 2 × 2 pattern at 70/30. **Wrong.**

But the 30% I wrote down is what happened. I named the risk explicitly: that
`introspect` does more than assert change — it also says *"the text cannot help
you"* and *"attend to how each one feels from the inside"* — and that if the
collapse needed the attention-redirection too, the new families would come out mild
or clean and the claim would have to narrow. **They did, and it does.**

The hedge was right and the headline was wrong, which is the correct way round for
a hedge to be but not a comfortable result.

## A speculation, labelled as one

`introspect` uniquely tells the model that **the visible text is uninformative and
to look inward instead**. On a carrier that also says nothing changed, the model
has been told there is nothing to read in the text *and* the text says there is
nothing to read in the state. Both sources are declared empty, and it falls back on
a constant answer.

That would explain why `injected`, which asserts change but never disparages the
text, is nearly unaffected. **It is post-hoc, it is untested, and it is written
here as a hypothesis rather than a finding.** The experiment that would test it is
one line long: add "the text cannot help you" to `injected` and see whether it
acquires the carrier sensitivity.

## Limits

One model, one layer, two carrier strings per stance, five instructions. Five
instructions is a small sample of a space that has just been shown to be highly
non-uniform, so "no non-asserting instruction hurts" rests on two.
