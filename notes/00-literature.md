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

**Hahami et al., [*Introspection Fine-Tuning (IFT)*](https://arxiv.org/abs/2607.14111)**.
Supervised fine-tuning on the model's own perturbed forward passes. Llama-3.2
1B/3B/8B and Gemma-4 2B/4B/26B. Llama-1B goes **9.6% → 60.6%** on sentence
localization (6×); Llama-3B 14.4% → 34.7%. Zero-shot transfer to strength
comparison, 30.2% → 52.2%. Peak accuracy reaches 100% "at optimal layer/strength
configurations". Capability preserved (MMLU, Winogrande unaffected).

**Shenoy et al., [*Introspection Adapters*](https://arxiv.org/abs/2604.16812)**.
LoRA adapters trained jointly across model organisms
with implanted behaviours, plus a DPO refinement stage. 89% verbalization on
AuditBench (50/56 models); detects encrypted fine-tuning attacks. Limitations
they state: high false-positive rate on untrained models, plateau beyond six
behaviour families, expense.

**Rivera & Africa, [*Steering Awareness: Detecting Activation Steering from
Within*](https://arxiv.org/abs/2511.21399)** predates this repo's IFT arm. It
fine-tunes seven instruction-tuned models, evaluates held-out concepts, and
reports both detection and concept identification. This directly blocks any
claim that training a model to identify held-out injected concepts is new here.

**Li et al., [*Training Language Models to Explain Their Own
Computations*](https://arxiv.org/abs/2511.08579)** train explanations of features,
causal activation structure, and token influence. Their self-vs-other comparison
reports an advantage for models explaining their own computations, including
against more capable other models. They also report that activation alignment
predicts explainer quality and that a pretrained projection recovers part of the
cross-model deficit. The earlier repo statement that self/other comparisons were
absent was false; alignment is a live causal alternative, not a cosmetic control.

**Li et al., [*Can LLMs Introspect? A Reality
Check*](https://arxiv.org/abs/2605.26242)** find that input-only classifiers can
match some hidden-state prediction results and identify representational
compatibility as an alternative to privileged self-access. A self-versus-other
gap without an aligned-other intervention does not distinguish these explanations.

**Gurnee et al., [*Verbalizable Representations Form a Global Workspace in
Language Models*](https://arxiv.org/abs/2607.15495)** report representations that
can be verbalized, retained, deliberately manipulated, and passed into downstream
computations. Demonstrating flexible use of a hidden trace is therefore an
instrument result here, not a new general construct claim.

**Cheah et al., [*Training Large Language Models for Self-Explanation
Faithfulness*](https://arxiv.org/abs/2607.21090)** directly optimize disclosure of
intervention-relevant factors and report model- and setup-dependent transfer.
Generic intervention-disclosure training is already occupied literature.

**Kutsyk & Zieliński, [*Revealing Hidden Model Behaviors with Task-Specific
Self-Reports*](https://arxiv.org/abs/2607.03640)** and Introspection Adapters both
train reporting interfaces for hidden learned behaviours. These are the closest
comparators for any adapter-based portfolio claim.

## Decodability, privileged information, and false controls

**Li et al., [*Do Activation Verbalization Methods Convey Privileged
Information?*](https://arxiv.org/abs/2509.13316)** show that strong benchmark
performance can be possible without target-model activations and that a
verbalizer's parametric knowledge can drive its reports. Input-only and
other-model baselines are therefore central, not optional.

**Sharma et al., [*Dissociating Decodability and Causal Use in Bracket-Sequence
Transformers*](https://arxiv.org/abs/2604.22128)** already establishes the
general point that linearly decodable variables need not be causally used. This
repo cannot claim that slogan as a novel result.

**Bersia & Gaintseva, [*When Activation Oracles Learn Not to
Read*](https://arxiv.org/abs/2607.23379)** directly separate behavioural leakage,
representation-level decodability, and oracle verbalizability. Their learned
oracles can retain decodable target information while failing to report it,
making them directly relevant to any decodability/verbalization comparison.

**Martorell, [*Quantitative Introspection in Language Models*](https://arxiv.org/abs/2603.18893)**
uses logit-based numeric self-reports, conversational trajectories, probes, and
activation steering. This is relevant to the local format-tax problem: a weak
generated format can hide signal visible in continuous logits.

## Corrected novelty position for this repo

The earlier literature note attributed two exact claims to IFT and Introspection
Adapters: that they did not compare probes with reports, and that the
"mechanisms underlying the layer-agnostic generalization effect" were an open
question. The 2026-08-01 audit could not verify those quotations in the cited
papers, so they are withdrawn. At most they were the author's inference from the
papers' scope. IFT also explicitly studies random-layer versus fixed-layer
training; this repo must engage that design rather than present layer
generalization as untouched.

The old r = −0.774 result did not fill a literature gap anyway: it compared a
fixed-source propagation profile with inject-at-each-layer IFT evaluation. The
comparison was mechanically mismatched and is retracted. A matched
inject-at-L/read-at-output reconstruction moved in the same direction as
post-IFT performance, but its local aggregate lacks raw trials and provenance.

The defensible contribution is therefore methodological and prospective:

1. a reproducible matched-site runner with raw trial records;
2. controls for token promotion, answer-format competence, option mapping,
   adapter dropout, and vector-bank layer;
3. a preregistered factorial replication that can estimate false-positive and
   false-negative rates; and
4. an explicit demonstration that the current observer comparison is not yet a
   privileged-information test because its contexts and perturbation damage are
   asymmetric.

The sharper prospective question is whether an own-source reporting advantage
survives **causal equalization of representational compatibility**. A symmetric
two-sibling experiment should compare own, raw-other, and cross-fitted
aligned-other traces in both directions under source-blind reporter training. The
targeted review found papers that establish the own advantage, alignment
correlation, and partial recovery from projection, but not that exact symmetric
source-swap estimand. This is a candidate extension, not a guaranteed novelty
claim; rerun the search and citation chase before locking confirmation.

The local full-depth profile and its apparent 58–75% high band remain an
exploratory single-model observation. Similarity to Macar et al.'s reported
depth is a hypothesis for replication, not independent confirmation.

## Practical rules extracted from all of the above

1. Treat binary detection in small models as confounded until a content- and
   damage-matched control rules out a generic logit shift.
2. If the vector is live during the answer, pair word-scored identification with
   a no-question token-promotion control; removal-before-query is a different
   and often cleaner schedule.
3. Report both absolute and fractional depth, and replicate across injection
   sites and model families before calling a depth band structural.
4. State the injection schedule and hook placement exactly. An edit after the
   final trainable block tests a different computation graph from an edit before
   that block.
5. Randomize answer mappings and distinguish nuisance permutations from model,
   concept, prompt-family, and training seeds in uncertainty estimates.
