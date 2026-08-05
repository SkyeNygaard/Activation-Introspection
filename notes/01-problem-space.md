# Problem space: what would count as introspection?

> **Audit note (2026-08-01):** this document described the original conceptual
> design. The observer arm is now treated as a stress test rather than a decisive
> privileged-access control; see `04-claim-audit.md` for the corrected design.

## The claim under test

A model is said to *introspect* when it reports on its own internal state using
access that is privileged — access an outside observer with the same inputs and
outputs would not have.

Almost every apparent demonstration of this is confounded, and the confound is
always the same shape.

## The one confound that matters

Inject a direction into the residual stream. The model says "I'm noticing
something about the ocean." Two explanations:

| | Mechanism | What it implies |
|---|---|---|
| **A. Introspection** | The model reads the perturbed state through some internal channel and describes it | Privileged access; interesting for safety |
| **B. Behavioural inference** | The injection biases the model's own token distribution toward ocean-ish text; the model notices its own drift and reports backwards from it | No privileged access; the model is just a good observer of its own output |

B is not a boring artefact. B is *the default hypothesis*, because it requires no
new machinery at all — the model already conditions on its own context.

Everything in this repo is organised around forcing A and B apart.

## Four separations

**1. Pre-verbalization report.** Ask the detection question *before* the model has
emitted any task text. This removes a generated behavioural transcript, but it
does not remove generic logit shifts or damage from the live intervention. It is
a useful diagnostic, not privileged-access evidence by itself. The v2 proposal
in `04-claim-audit.md` removes the hook before revealing the answer codebook.

This is the single most important condition. It is also fragile: the chat
template, the prefill, and the position mask must guarantee that no injected
token has been emitted yet. `Intervention(positions="all")` on the prompt is fine;
`positions="generated"` is what makes the pre-verbalization arm meaningful.

**2. The observer arm.** Take the intervened model's *output* and hand it to a
clean copy of the same model with `OBSERVER_IDENTIFY`. The observer receives one
candidate behavioural channel, but not the introspector's exact transcript or
damage state. Report the pair, while keeping that asymmetry explicit:

- introspector accuracy ≫ observer accuracy → evidence for A
- introspector accuracy ≈ observer accuracy → B fully explains the result
- observer > introspector → see below

Publishing the introspector number alone is the field's most common overclaim.

> **Correction (2026-07-31, after the first powered sweep).** I originally read
> "observer > introspector" as meaning the elicitation prompt is the bottleneck.
> That is incomplete, and the completion matters.
>
> The comparison is **not symmetric**. The injection that produces the behavioural
> signal the observer reads is the same injection that degrades the introspector's
> ability to answer anything. The observer reads a transcript using an undamaged
> model; the introspector is answering while perturbed. So the observer enjoys a
> structural advantage that grows with injection strength.
>
> The gap is therefore **biased against introspection**, which means a null gap is
> weaker evidence than it appears and a positive gap is stronger. At 0.5B the
> measured gap was −0.072 [−0.117, −0.031] — negative and excluding zero.
>
> Matching behavioural KL is one useful repair, but it does not remove the
> different-context or clean-observer advantages. The 2026-08-01 clustered
> reanalysis also moved the interval to [−0.236, 0.047], so the old IID interval
> is historical rather than inferential evidence.

**3. Dissociating magnitude from behavioural effect.** Sweep injection strength.
Two curves:

- report accuracy vs **injection norm** (the internal quantity)
- report accuracy vs **behavioural effect size** — e.g. KL between intervened and
  clean next-token distributions on a neutral task

Under a narrow leakage-only version of B, accuracy should largely track
behavioural effect. So find
pairs of conditions with *matched* behavioural effect but different injection
geometry (different layers, or a concept direction vs a shuffled control tuned to
equal KL). A difference is a useful diagnostic, but geometry can change leakage
in ways a scalar next-token KL does not capture, so it is not decisive by itself.

**4. Matched-norm null directions.** `random_control` and `shuffled_control`
inject something with the same magnitude and no meaning. If the model confidently
names a concept under those, its reports are confabulation with a detection
threshold, not readout. Measure the *false alarm rate*, and report detection as
AUROC over (injected, not-injected) rather than as raw hit rate — a model that
always says YES scores 100% on hit rate.

## Why layer and position are the real experiment

Introspective access, if it exists, should have structure:

- **Layer profile.** Injection too early gets washed out or reinterpreted as
  input; too late leaves no compute to notice it. A hump-shaped accuracy-vs-layer
  curve is what a genuine mechanism looks like. A flat or monotone curve usually
  means you are measuring behavioural leakage.
- **Position.** Injecting on the prompt lets the model treat the perturbation as
  part of the question. Injecting only on generated tokens is a cleaner test of
  "notice a change in yourself mid-thought."

Normalising strength to the measured residual norm at that layer (`normalize=True`)
is not a nicety. Residual norm grows several-fold with depth; a fixed alpha across
a layer sweep produces a monotone curve that is entirely an artefact of scale.

## Scale caveat, stated up front

Introspective report is plausibly emergent. A 1.5B instruct model may fail every
condition, and that failure would say little about frontier models. The honest
framing for a laptop-scale artifact:

> This establishes the measurement apparatus and the control structure, validated
> on small models, and reports where the effect first appears across the scales I
> could run. It does not claim a frontier-model result.

Running `qwen-0.5b → 1.5b → 7b` and showing the controls behave sensibly at each
size is a better artifact than a single unreplicated number at one scale.

## Falsifiable predictions worth writing down before running anything

1. Detection AUROC at matched-norm random control ≈ 0.5 (else the model is
   pattern-matching on "perturbed at all", not on content).
2. Identification accuracy is hump-shaped in layer, peaking in the middle third.
3. Observer accuracy rises with injection strength faster than introspector
   accuracy does, because strong injection makes the output obviously themed.
4. Pre-verbalization identification is at chance for models below ~7B.

Prediction 4 being wrong is the interesting outcome. The observer gap is now a
secondary diagnostic rather than the headline until transcript and damage are
matched.

## Open threads

- Does the model's *confidence* track anything real, or is it flat?
- Ablation rather than addition: can the model report a concept being *removed*?
  Harder to explain via B, since ablation often has a small behavioural signature.
- Does a model report interventions on *itself* better than interventions on a
  different model's stream presented as its own? Prior work already includes
  self/other comparisons (notably arXiv 2511.08579); a local version would be a
  replication/extension, not a first.
