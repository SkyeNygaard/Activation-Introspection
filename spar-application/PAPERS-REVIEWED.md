# Papers reviewed

A running ledger of what has actually been read, how deeply, and what it did to
this repository's claims. Started **2026-08-12** because the same papers were
being re-found and re-argued across sessions.

**Depth is recorded honestly**, because "we looked at that one" is worth nothing
without knowing whether anyone opened it:

| depth | means |
|---|---|
| **full** | read end to end, or every section that bears on a claim here |
| **targeted** | abstract, plus the specific sections that decide a claim |
| **abstract** | abstract and headline numbers only |
| **listed** | found in a search, not opened |

PDFs are downloaded to `papers/` at the repository root. **That directory is
gitignored** — 34 MB of PDFs would triple the clone size, and the point of this
file is that the reading is recorded, not that the files ship.

---

## Core introspection line

| paper | depth | date read | what it settles here |
|---|---|---|---|
| [Emergent Introspective Awareness in LLMs](https://arxiv.org/abs/2601.01828) (Lindsey) | targeted | 2026-08-12 | The paradigm everything here builds on. **Uses norm-matched random vectors as a control and finds base models are selective** — random vectors need a larger norm and still only reach 9 trials in 100. So the "untrained model is selective" half of [notes/08](../activation-introspection/notes/08-sensitivity-specificity-tradeoff.md) is **prior art**. Also flags that identification may be confabulated while detection is genuine |
| [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396) (Macar, Yang, Wang, Wallich, Ameisen, Lindsey) | full | 2026-08-12 | Current frontier. Two-stage circuit: evidence carriers suppress a default-"No" gate. Capability comes from **post-training, specifically DPO and not SFT**; absent in base models. **Introspection is underelicited** — refusal ablation lifts detection 10.8% → 63.8% (false positives 0% → 7.3%); a trained bias vector gives +75% on held-out concepts. **Their false-positive control is unsteered trials — nothing injected.** Their Responsible Use section recommends "side-effect audits" they did not run. This is the open end [notes/08](../activation-introspection/notes/08-sensitivity-specificity-tradeoff.md) speaks to |
| [Emergent Introspection in AI is Content-Agnostic](https://arxiv.org/pdf/2603.05414) (Lederman, Mahowald) | full | 2026-08-12 | **Publishes [notes/23](../activation-introspection/notes/23-held-out-semantic-generalization.md)'s conclusion first.** Models detect that something was injected without knowing what: 74.8% of one model's 4,733 wrong guesses are the word "apple", and confabulations are reliably more concrete, positive and frequent than the injected concept. **Not in tension with [notes/14](../activation-introspection/notes/14-content-versus-disturbance.md)** — they score open-ended naming, notes/14 scores forced choice between arbitrary labels. Their §2 names the steering-favours-the-answer-token confound that this repository's re-randomised labels remove by construction |
| [Introspection Adapters: Training LLMs to Report Their Learned Behaviors](https://arxiv.org/pdf/2604.16812) (Anthropic) | targeted | 2026-08-12 | **The closest prior art to [notes/08](../activation-introspection/notes/08-sensitivity-specificity-tradeoff.md).** Limitations, first sentence: adapters "exhibit a high false positive rate: when applied to models without the specific behaviors seen during training, they tend to hallucinate behaviors from the training distribution", and reducing it is "an important direction for future work". Different object (learned behaviours, not injected activations) and a weaker control (absence of behaviour, not a matched-magnitude meaningless one), but **the direction of notes/08 is published** |
| [Introspection Fine-Tuning (IFT)](https://arxiv.org/abs/2607.14111) (Hahami et al.) | targeted | 2026-08-12 | Binary detection is confounded in small models (r = 0.999 with a factual-no control); proposes localization and strength-comparison instead. Its Gaussian arm is a **training-data** variable — train on noise vs concepts — **not** a test-time control, so it does not pre-empt notes/08's estimand |
| [Detecting the Disturbance / Feeling the Strength but Not the Source](https://arxiv.org/abs/2512.12411) (Hahami et al.) | abstract | 2026-08-05 | Binary detection conflates introspection with global logit shifts; real partial introspection on tasks needing differential sensitivity, confined to early layers. Matches the layer-9 site used here |
| [Steering Awareness](https://arxiv.org/abs/2511.21399) (Rivera, Africa) | abstract | 2026-08-05 | Held-out concepts, vector-construction transfer, layer/position effects, distributed transformation. 7B model trained from 0.4% to 85% on held-out concepts, false positives 6.7% → 0% — again against a no-injection control |
| [Latent Introspection: Models Can Detect Prior Concept Injections](https://arxiv.org/html/2602.20031) (Pearson-Vogel et al.) | abstract | 2026-08-05 | Model denies injection in its output while a lens shows detection in the residual stream. Independent convergence with this repository's readable-but-unused headline, by a different measurement |

## Comparator, cost criterion, and privileged access

| paper | depth | date read | what it settles here |
|---|---|---|---|
| [Training Language Models to Explain Their Own Computations](https://arxiv.org/abs/2511.08579) (Li, Guo, Huang, Steinhardt, Andreas) | full | 2026-08-12 | **Belinda Li's paper; the target of project 1's critique question.** Privileged Access Hypothesis: a model explains itself better than other models explain it. Three tasks, ~100× more sample-efficient than nearest-neighbour. **Their comparator is another language model, both given activations** — not a cost-matched cheap reader. They report explainer quality tracks representational similarity, which is the mundane reading and is in their own results. **No draft critique text exists in this repository, deliberately** |
| [The cost criterion](https://arxiv.org/abs/2508.14802) | abstract | 2026-08-05 | The operative definition this repository tests against: introspection requires beating an equal-or-lower-cost third party |
| [Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242) | abstract | 2026-08-05 | Input-only classifiers match some hidden-state prediction results; names representational compatibility as the alternative to privileged self-access |
| [Steerable but Not Decodable](https://arxiv.org/html/2604.02608v2) | abstract | 2026-08-12 | Function-vector steering succeeds where the logit lens decodes nothing, 4,032 pairs across 12 tasks and 6 models. **[notes/18](../activation-introspection/notes/18-where-the-lens-fails.md) is a single-model instance of this** |
| [Looking in the Mirror](https://arxiv.org/html/2608.04347) | targeted | 2026-08-12 | Side-effect introspection: can a fine-tuned model report *unintended* alignment degradation. Prior art for [notes/12](../activation-introspection/notes/12-training-versus-a-probe.md). Also the clearest statement of why the field wants self-report as a cheap audit — which is what makes notes/08's precision cost matter |
| [Introspective Coupling](https://arxiv.org/abs/2606.32038) | listed | — | Explanations track counterfactual behaviour despite fixed supervision |
| [Do Activation Verbalization Methods Convey Privileged Information?](https://arxiv.org/abs/2509.13316) | listed | — | Some verbalization methods convey no privileged information |

## Found in the 2026-08-12 deep search, not yet opened

Recorded so they are not re-found from scratch. **All `listed` — nobody has read
these.**

| paper | why it might matter |
|---|---|
| [Revealing Hidden Model Behaviors with Task-Specific Self-Reports](https://arxiv.org/html/2607.03640) | Self-report as an auditing signal; adjacent to notes/08's monitor framing |
| [Can LLMs Reliably Self-Report Adversarial Prefills, and How?](https://arxiv.org/pdf/2606.23671) | Self-report under adversarial input — the adversarial case notes/08 gestures at |
| [Predictive Concept Decoders](https://arxiv.org/pdf/2512.15712) | Trained end-to-end interpretability assistants; a comparator tier |
| [The Signs Were Always There](https://arxiv.org/pdf/2606.12629) | Training-free concept detection in raw dimensions; bears on the cheap-reader ladder |
| [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al.) | **The method any refusal-ablation experiment here would use.** Must be read before proposing one |
| [Mechanisms of Introspective Awareness — code](https://github.com/safety-research/introspection-mechanisms) | Their appendices/code are where a matched-magnitude control would hide if one exists |

## Selective prediction — the literature crossed in on 2026-08-12

Searched because [notes/29](../activation-introspection/notes/29-can-abstention-recover-selectivity.md)
imports this machinery. **Two separate clusters with nothing joining them**, which
is the gap notes/29 occupies.

| paper | depth | what it establishes |
|---|---|---|
| [Calibrating LLMs for Selective Prediction](https://openreview.net/forum?id=JJPAy8mvrQ) | abstract | Risk-coverage optimisation for LLM *task answers*. The standard framing |
| [Uncertainty-Aware Abstention with Provable Guarantees](https://arxiv.org/pdf/2607.04430) | listed | Abstention with alignment guarantees. Task answers again |
| [LLM Abstention Can Be a Prompt Artifact](https://arxiv.org/pdf/2507.16199) | listed | Abstention may reflect prompt framing rather than genuine uncertainty — a direct caution for any abstention result, including notes/29 |
| [Same-Model Self-Verification as a Conditional Confidence Signal](https://arxiv.org/pdf/2605.02915) | listed | Self-verification improves risk-coverage. Closest in spirit; still about answers, not about self-reports of internal state |

**Nothing found applies selective prediction to introspective self-reports.**
Every introspection paper in the table above reports accuracy and false-positive
rate at full coverage. That is the opening notes/29 uses, and it is also why the
result should be labelled an extension: the tool is standard, only the target is
new.

## What the 2026-08-12 search did and did not do

**Did:** nine papers fetched and read to the depths above; an arXiv API sweep
across eight query formulations; roughly ten targeted web searches across
distinct phrasings; direct inspection of the limitations and control sections of
the four closest papers.

**Did not:** systematic backward and forward citation chaining (the Semantic
Scholar API rate-limited the attempt); appendix and code inspection of
Introspection Adapters and Mechanisms of Introspective Awareness; the mentors'
own reading lists.

Under [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md)'s five-step rule this is
**step 3 of 5**. Nothing in this file is a novelty claim. It is a record of what
was read and what it decided.

## The rule this ledger exists to enforce

Search **before** running, not after. Every search in this repository so far has
happened after the experiment, in violation of
[LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md)'s own rule, and each one has
found prior art the work did not expect. Two results have been demoted this way
on a single day.
