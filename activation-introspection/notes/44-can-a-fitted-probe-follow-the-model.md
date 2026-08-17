# 44 — Can a fitted probe follow the model to the answer? (pre-run note)

Queued **2026-08-17**. Not run at the time of writing.

## The lead this is built on, and why nothing was built on it before

[notes/18](18-where-the-lens-fails.md), 2026-08-12, found something and then the
project walked past it for five days:

> At the position the model actually answers from, the concept is at **chance —
> 0.125 — for blocks 9 through 27**, nineteen consecutive blocks, while the model
> identifies it at 0.667. […] **This is the first result here that looks like the
> model doing work a cheap reader cannot follow.**

Every other result in this repository points the other way: hand an outside method
the same state and it wins. This is the one cell where it does not.

**It was never turned into a claim because of one obvious objection.** The reader in
notes/18 is a *logit lens* — the model's own unembedding, an unfitted readout. Anyone
would say: of course an unembedding read fails mid-network, it is looking for the
concept in token space. **Fit a probe.** Nobody has.

That single unrun check is the whole distance between a curiosity and a result.

## What I am about to do

Capture the residual state at both sites — the injection marker and the position the
model answers from — at every depth, and fit a cross-validated probe on each. Compare
against the model's own forced choice on the identical episodes.

| | |
|---|---|
| model | Qwen2.5-3B, inject at layer 9, strength 2.0 — unchanged from `18` |
| episodes | **88** — 8 concepts × 11 carriers, against `18`'s 24 |
| carriers | the 3 originals plus 8 written blind for [41](41-how-often-does-the-prompt-backfire.md) |
| depths | every block from 9 to 35 |
| sites | `marker` (where the edit lands) and `final` (where the answer is formed) |
| probe | `introspect.probe.fit_probe` — cross-validated multinomial logistic, never scores a trial it trained on |
| also measured | the logit lens at each cell, and the model's own accuracy on the same 88 episodes |

**Why 88 and not 24.** `18` had three episodes per concept. A probe cannot be fitted
and held out on that. Eleven carriers gives eleven per concept, and today's
[43](43-the-ladder-at-proper-power.md) showed carrier identity moves the model tier
by three and a half fold, so a single-carrier estimate would be worthless anyway.

## The controls, declared before the run

1. **A shuffled-label null**, which `fit_probe` provides. In 2048 dimensions a probe
   can separate almost anything — that is precisely the artifact
   [38](38-identity-or-displacement.md) caught, where 39 points in 2560 dimensions
   gave a spurious 1.000. **If the shuffled null is not at chance, the cell is
   uninterpretable and I say so rather than reading the real number.**
2. **The anchor.** The lens at the marker site, block 9, must reproduce `18`'s
   **1.000**. If it does not, the harness moved and nothing else is readable.
3. **The probe is given every advantage.** It gets supervised labels the model never
   sees, and a random cross-validation split rather than a by-carrier one. That is
   deliberately generous: if it *still* cannot read the answer position, the result
   is much harder to argue with.

## What each outcome means, including the boring one

| result | reading |
|---|---|
| the fitted probe is at chance at the answer position where the model is at 0.667 | **The result this project has never had.** Information the model uses about its own state that a fitted, supervised, cost-matched external readout cannot recover from the position the answer is formed. It is the black-box-relevant regime, and it inverts this repository's headline from "the outsider always wins" to "here is exactly where it loses" |
| the probe recovers it, lens does not | `18`'s finding was an artifact of using an unfitted readout. The comparator story stands unchanged, the last upward lead closes, and nobody should cite `18` as evidence of anything the model uniquely does |
| the probe recovers it only in late blocks | A depth boundary: the information is not linearly present at the answer position until block *k*. Weaker but real, and it names where to look |
| the shuffled null is above chance anywhere | That cell is overfitting and is reported as void, not as a result |

**The second row is the boring outcome and it is the one I should expect to want to
argue with.** It closes the only lead in this repository pointing at a positive
finding, which is exactly the condition under which I would be tempted to rescue it.
Declared now: if the probe reads the answer position, `18` is downgraded to "the
unembedding is a weak readout" and no further variant is run.

## Prediction, on the record

**I expect the probe to recover the concept at the answer position, and `18`'s
finding to be an artifact of the unfitted lens — about 65/35.**

Reason: the model demonstrably uses the information at that position, so it is
present in the state in *some* form, and a supervised linear probe with 88 examples
is a much stronger instrument than an unembedding read. The honest counter, and why
it is not 90/10: "the model uses it" does not mean "it is linearly decodable" — a
representation can be present and non-linear, and notes/18's own account is that the
concept stops being *token-aligned* on the way, which is not the same as stopping
being linearly present.

**So I am predicting against the outcome this project needs.** If I am wrong, the
result is the strongest thing here.

## What it costs

88 forward passes with capture at 27 depths, inference only, one model load. Probe
fitting is CPU and instant. Roughly **five minutes**. Smoke on one carrier first.
