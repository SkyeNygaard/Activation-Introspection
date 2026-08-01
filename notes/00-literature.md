# Where the introspection literature is, as of August 2026

Read before extending this repo. Written after discovering that two of my own
"findings" were already published — the check costs twenty minutes and should
precede the experiment, not the writeup.

## The established result

**Lindsey (2026), *Emergent Introspective Awareness in LLMs*** (arXiv 2601.01828,
transformer-circuits.pub). Introduced the concept-injection paradigm: inject a
steering vector into the residual stream, ask the model whether it detects an
injected thought and what it is about. Claude Opus 4/4.1 reach ~20% introspection
rate at ~0% false positives.

Methodological detail that matters and that I initially failed to replicate: the
vector is injected **during the KV-cache generation of an initial turn and removed
before the model is queried**. That design choice is what prevents the injection
from mechanically promoting the concept's own token in the answer. Injecting
*during* the answer — what I did first — makes any word-scored identification
circular.

## The confounds, now well characterised

**Krasheninnikov et al., *Detecting the Disturbance*** (arXiv 2512.12411). Binary
yes/no detection in small models is **entirely** a global logit shift. Their
control: apply the identical injection to factually-false questions ("Can humans
breathe underwater?"). The affirmative shift is the same, r = 0.999. Detection
accuracy in small models is not evidence of introspection.

Crucially they also show what *does* survive: **sentence localization** (which of
N sentences was perturbed) reaches ~88% against 10% chance, and **strength
comparison** ~83% against 50%. So the confound is specific to the binary
paradigm, not to introspection tasks generally.

**Anon., *Emergent Introspection in AI is Content-Agnostic*** (arXiv 2603.05414).
Models detect that an anomaly occurred while failing to identify *what* it was,
confabulating toward common concrete nouns ("apple"). Detection requires fewer
tokens than identification, and wrong guesses arrive earlier — a temporal
dissociation suggesting distinct mechanisms for noticing and for naming.

**Macar et al. (2026), *Mechanisms of Introspective Awareness*** (arXiv
2603.21396). Locate a distributed introspective circuit at roughly **70% of model
depth** in Gemma3-27B and Qwen3-235B. Causal pathway: injected concepts activate
evidence carriers in early post-injection layers, which suppress late-layer gate
features that otherwise promote the default "No". Introspection is a non-linear
computation that emerges from post-training.

## Training models to introspect — the active frontier

**Hahami et al., *Introspection Fine-Tuning (IFT)*** (arXiv 2607.14111).
Supervised fine-tuning on the model's own perturbed forward passes. Llama-3.2
1B/3B/8B and Gemma-4 2B/4B/26B. Llama-1B goes **9.6% → 60.6%** on sentence
localization (6×); Llama-3B 14.4% → 34.7%. Zero-shot transfer to strength
comparison, 30.2% → 52.2%. Peak accuracy reaches 100% "at optimal layer/strength
configurations". Capability preserved (MMLU, Winogrande unaffected).

**Anthropic, *Introspection Adapters*** (arXiv 2604.16812,
alignment.anthropic.com). LoRA adapters trained jointly across model organisms
with implanted behaviours, plus a DPO refinement stage. 89% verbalization on
AuditBench (50/56 models); detects encrypted fine-tuning attacks. Limitations
they state: high false-positive rate on untrained models, plateau beyond six
behaviour families, expense.

## The gap this repo targets

Both training papers state they do **not** compare what a linear probe can decode
from activations against what the model verbalizes:

- IFT "does not employ linear probes to compare what activations encode versus
  what models report", and lists "mechanisms underlying the layer-agnostic
  generalization effect" as open.
- Introspection Adapters "don't explicitly compare linear probes versus
  verbalization capabilities", and state the open question as *why* the adapters
  generalize across differently-trained models.

So nobody has measured **how much concept information is linearly present before
any introspection training** — which is precisely the quantity that should
determine how much such training can gain.

That is what `scripts/layer_profile.py` measures, and it yields a falsifiable
prediction: **introspection-training gains should track pre-training transfer-probe
decodability, layer by layer.** Where transfer sits at the permuted-label null,
training should have nothing to latch onto; where transfer is high, gains should
be large. If the IFT layer optima coincide with the decodability profile measured
here, that is a mechanistic account of when introspection training works — and a
way to predict where it will fail without running the training.

Supporting convergence: Macar et al. put the introspective circuit at ~70% depth;
the transfer profile on Qwen2.5-0.5B rises into its stable high band at 58–75%
depth (layers 14–18 of 24).

## Practical rules extracted from all of the above

1. Never report binary detection accuracy in a small model. It is a logit shift.
   Use localization or strength comparison, or report AUROC with a matched null.
2. Never score identification on tokens the injection promotes, and if the vector
   is live during the answer, run the no-question control.
3. Report the layer as a fraction of depth, not an index — the interesting
   structure sits at 60–75% across very different models.
4. State the injection schedule explicitly (during-prompt, during-answer, removed
   before query). It changes what the experiment measures.
