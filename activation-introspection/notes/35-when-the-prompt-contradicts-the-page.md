# 35 — When the prompt contradicts the page (pre-run note)

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

---

# Result: confirmed, confined to one cell, and it is an instance of something
# already known

Run **2026-08-13**, 576 episodes, 48 twin cells per condition. Artifacts:
`results/prompt_clash_v1_raw.jsonl`, `results/prompt_clash_v1_summary.json`.

| carrier | instruction | constant-label | twin-pair | accuracy | mean confidence |
|---|---|---:|---:|---:|---:|
| `denies` | `silent` | **0.000** | **1.000** (48/48) | 1.000 | 7.43 |
| **`denies`** | **`asserts`** | **0.875** | **0.125** (6/48) | 0.562 | **10.72** |
| `neutral` | `silent` | 0.208 | 0.792 (38/48) | 0.896 | 4.51 |
| `neutral` | `asserts` | 0.104 | 0.896 (43/48) | 0.948 | 7.64 |
| `affirms` | `silent` | 0.062 | 0.938 (45/48) | 0.969 | 5.66 |
| `affirms` | `asserts` | 0.146 | 0.854 (41/48) | 0.927 | 8.80 |

Effect of adding the instruction, per carrier:

| carrier | change | 95% CI |
|---|---:|---|
| `denies` | **−0.875** | **[−0.958, −0.771]** |
| `neutral` | +0.104 | [−0.042, +0.250] |
| `affirms` | −0.083 | [−0.208, +0.042] |

**Interaction: −0.885, 95% CI [−1.010, −0.750].**

## What this establishes

**The collapse is confined to exactly one cell, and it is the predicted one.**
48 of 48 becomes 6 of 48 where the instruction contradicts the page. Where the
carrier is neutral the same instruction is mildly *helpful*; where the carrier
agrees with it, mildly harmful and nowhere near significance.

Three alternatives are ruled out:

- **"That carrier is just hard."** It is the *easiest* — 48/48, zero constant
  labelling, without the instruction.
- **"That carrier is short."** `affirms` is the same length and register and shows
  no collapse. Stance, not length.
- **"The instruction is simply bad here."** It helps on `neutral` and costs 0.083
  on `affirms`.

**And confidence rises in every `asserts` cell** — 7.43 → 10.72, 4.51 → 7.64,
5.66 → 8.80. The instruction reliably makes the model more certain. On the
contradicting carrier it makes it more certain while destroying the readout. The
model is not confused; it is confidently answering a different question.

My prediction was collapse confined to `denies` × `asserts` at 70/30, with
`affirms` fine. **Confirmed on every particular**, including the length control I
was least sure of.

## The honest novelty position, checked before claiming

**Instruction–context conflict degrading model behaviour is well-studied prior
art.** Searched 2026-08-13:
[Three Regimes of Context-Parametric Conflict](https://arxiv.org/html/2605.11574),
[Task Competence Is Not Instruction Following](https://arxiv.org/html/2607.19608)
on small models failing when instructions conflict with task behaviour, and
[Instruction-Tuned LMs Cannot Sample from Distributions They Can Describe](https://arxiv.org/html/2607.25292v1)
on instruction tuning amplifying collapse to a single output. Semantically
coherent distractor text collapsing accuracy is also documented.

**So the phenomenon is not new and must not be presented as new.** What this adds
is narrower and is a methodological caution rather than a discovery:

> An **introspection elicitation prompt** is subject to it, the failure mode is a
> confident collapse to one label rather than a visible degradation, and it is
> invisible in the pooled averages this literature reports its gains as.

The prompting boosts in the introspection literature are single numbers over a set
of inputs — 0.3% → 39.9%, 10.8% → 63.8%. This is a worked example of a prompt that
would show a gain on average while silently taking one input class from perfect to
below chance. Nobody in that literature reports a per-input breakdown, and this is
a concrete reason to.

**Label: extension, and a caution.** The general effect is published; pointing it
at introspection elicitation, with a controlled stance × stance design and the
constant-label diagnostic, is what is added.

## What it does *not* establish

One model, one instruction family, one carrier per stance, four concept pairs. One
carrier per stance is the sharpest limit: `denies` is a single string, so "carriers
that deny change" is a class of one, and the effect could belong to that sentence
rather than to its stance. A second string per stance is the obvious next control
and it is cheap.

Nor does this show the effect exists at scales where the introspection prompting
results were obtained. It is a 3B model.

## Where this leaves the line

`34` said the 29→31→32→33 programme stops. This closes the loose end it left: the
clash is measured rather than inferred, and it is prior art in general form. So
the branch is finished rather than merely paused, and the durable outputs are
`29` and `31` on abstention, `32`'s scale boundary, and this caution.
