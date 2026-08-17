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
| [Latent Introspection: Models Can Detect Prior Concept Injections](https://arxiv.org/html/2602.20031) (Pearson-Vogel et al.) | **targeted 2026-08-12** (was abstract, 2026-08-05) | 2026-08-12 | Two things. **(a)** Model denies injection in its output while a lens shows detection in the residual stream — convergence with this repository's readable-but-unused headline. **(b) It is prior art for [notes/21](../activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md), which the abstract-depth read missed entirely.** Their *first* listed contribution is a systematic prompting sweep: 16 conditions, and asking plainly moves detection from 0.3% while adding mechanism detail moves it to 39.9%, with prompting eliciting up to 84.0% accuracy. "Detection accuracy varies dramatically across prompting conditions" is their figure caption. notes/21's 2.4× range over six prompts is the same claim, smaller and less systematic, five months later. **Reading at abstract depth cost this**, which is the argument for this file existing |

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

## Flagged by an outside review 2026-08-14, opened 2026-08-17

An independent review named these as directly relevant and missing. They have now
been read, at the depths recorded. **The headline: the codebook-ICL paradigm has
closer prior art than this repository recorded, and it does not sink the branch —
but it removes two things that were being treated as ours.**

| paper | depth | date read | what it settles here |
|---|---|---|---|
| [Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations](https://arxiv.org/abs/2505.13763) (Ji-An, Zhang, Mattar, Fang, Lee, Xiao et al.) | targeted | 2026-08-17 | **The closest prior art to the codebook-ICL branch, and the reviewer was right to flag it.** A "neuroscience-inspired **neurofeedback** paradigm that uses **in-context learning**": N sentence–label pairs in dialogue turns, where each label is the model's *own* activation projected onto a target axis and discretized. LLaMA-3 (1B–70B) and Qwen2.5 (1B–7B), five depths at the 0/25/50/75/100 percentiles, 600 examples to fit the axis and 600 held out. **Prior art for three things.** (a) In-context learning of an activation→label mapping works. (b) **Semantic interpretability of the axis drives performance** — logistic-regression axes "outperform" principal-component axes — which is [notes/14](../activation-introspection/notes/14-content-versus-disturbance.md)'s contrast in related form, published first. (c) **Variance explained by the axis drives performance** — "earlier PCs being reported more accurately" — which is adjacent to the clustering→learnability line. They also do **control**, not just reporting, which this repository has never attempted. **What is not theirs:** the direction is *read* where it naturally falls, never causally injected; there is no byte-identical twin control and no baseline pinning a text-only strategy at chance by construction (their device is a minimal "Say something" prompt); the label mapping is not re-randomised per episode; and their comparator is a theoretical "ideal observer" with perfect access, not a cost-matched cheap reader |
| [Privileged Self-Access Matters for Introspection in AI](https://arxiv.org/abs/2508.14802) (Song, Lederman, Hu, Mahowald) | targeted | 2026-08-17 | The criterion this repository tests against, now read rather than cited. The definition is **computational** cost, not wall-clock cost — footnote 2 says so explicitly — and a four-shot centroid comfortably clears that bar against a 3B forward pass, so the comparator experiments are using the criterion correctly. **Two things to carry.** The third party is specified as "without special knowledge of the situation", and the paper explicitly contemplates that third party running "a computationally intensive probe" — so a *cheap* probe beating the model is exactly the criterion's negative case, as used here. But footnote 2(ii) volunteers that a state may be **too low-level to count**: "if a model has a shortcut to ascertain the value of one neuron very efficiently, intuitively this would not count as introspection". An injected concept direction at one layer is open to that objection and the application should raise it rather than wait for it |
| [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/abs/2410.13787) (Binder et al.) | abstract | 2026-08-17 | **A positive privileged-access result, on a different object.** M1 predicts its own behaviour better than M2 does, *even when M2 is trained on M1's outputs*, and it survives deliberately altering M1's behaviour — so it is not memorisation. Fails on complex and out-of-distribution tasks. **Consequence for this repository's headline:** the negative result here is about *activations at one site*, and must not be stated as "models have no privileged access". Behavioural self-prediction is a live positive case and this repository has not tested it |
| [Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions, and Improve with Training](https://arxiv.org/abs/2505.17120) (Plunkett et al.) | abstract | 2026-08-17 | GPT-4o and 4o-mini fine-tuned on decisions driven by randomly generated preference weights can report those weights, training improves it, **and the improvement generalizes to decision types not fine-tuned on**. That is the positive version of the generalization question notes/23 answers negatively — on learned decision policies rather than injected activations. No numbers obtained at this depth; **open it in full before citing the generalization claim** |
| [Language Models Fail to Introspect About Their Knowledge of Language](https://arxiv.org/abs/2503.07513) (Song, Hu, Mahowald) | abstract | 2026-08-17 | 21 open models, two linguistic domains. Metalinguistic prompted answers do **not** predict the model's own string probabilities, controlling for models with near-identical internal knowledge. Supports this repository's direction: high task accuracy from prompting is not evidence of self-access. A useful citation for why the anchor condition scoring 0.875 does not rescue the held-out failure |

### What this changed, stated plainly

1. **"Models underperform an observer given their activations" is prior art.** Ji-An
   has it against an ideal observer. What remains this repository's is the *cost-matched
   cheap* reader and the twin construction that makes the comparison interpretable.
2. **[notes/14](../activation-introspection/notes/14-content-versus-disturbance.md)'s
   demotion on 2026-08-14 was correct, and now has an external reason too.** Semantic
   interpretability of an axis driving reportability is published.
3. **The clustering→learnability line stays dead.** It failed its own replication in
   [notes/26](../activation-introspection/notes/26-someone-elses-rules.md), and Ji-An's
   variance-explained finding is adjacent prior art. Two independent reasons.
4. **The strongest thing this repository holds is a control, not a result** — the
   byte-identical twin pinning an input-only strategy at exactly 0.500 by construction.
   None of the five papers has an equivalent. Lead with it.
5. **A path nobody here has touched:** Ji-An reports *control* as well as monitoring —
   a model changing its own activations on request. Also untouched: behavioural
   self-prediction (Binder), where the privileged-access answer is positive.

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

## Instruction–context conflict — searched 2026-08-13 for notes/35

| paper | depth | what it settles here |
|---|---|---|
| [Three Regimes of Context-Parametric Conflict](https://arxiv.org/html/2605.11574) | listed | Conflict between parametric knowledge and provided context, with a predictive framework |
| [Task Competence Is Not Instruction Following](https://arxiv.org/html/2607.19608) | listed | Small models fail to comply when instructions conflict with their usual task behaviour |
| [Instruction-Tuned LMs Cannot Sample from Distributions They Can Describe](https://arxiv.org/html/2607.25292v1) | listed | Instruction tuning amplifies collapse to a single output |

**Verdict: instruction–context conflict degrading behaviour is prior art**, so
[notes/35](../activation-introspection/notes/35-when-the-prompt-contradicts-the-page.md)
must not be presented as a new phenomenon. What it adds is the application —
an *introspection elicitation* prompt is subject to it, the failure mode is a
confident collapse to one label, and it is invisible in the pooled averages this
literature reports gains as. Extension and caution, not discovery.

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

## The 2026-08-17 pass, and the rule it broke on the way

Five flagged papers opened; the codebook branch's closest prior art identified and
its consequences entered above. Still not done: citation chaining, appendices and
code, the mentors' reading lists. **Still step 3 of 5.**

**Recorded because the file exists to record this:** [notes/39](../activation-introspection/notes/39-what-does-the-model-actually-use.md)
was designed, queued and run *before* this search, which is the exact ordering this
ledger's closing rule forbids. It cost nothing this time — the search found no prior
art bearing on notes/39's specific manipulation, and the result was a null anyway —
but that is luck, not process. The search happened after the experiment for the
fourth time in this repository.

## The rule this ledger exists to enforce

Search **before** running, not after. Every search in this repository so far has
happened after the experiment, in violation of
[LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md)'s own rule, and each one has
found prior art the work did not expect. Two results have been demoted this way
on a single day.
