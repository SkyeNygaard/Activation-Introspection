# Literature and novelty boundary

Last checked: **2026-08-12** for notes 11–16; **2026-08-09** for everything
earlier. This is a targeted audit of the closest primary papers and the six
official project descriptions, not a systematic review. A fresh search, citation
chase, and mentor-provided reading list are required before locking any claim of
novelty.

> **Rule 5 of this file was broken, and it is recorded rather than repaired
> quietly.** The rule says to append new papers *before* confirmatory data are
> inspected. Notes 11 and 12 ran on 2026-08-11 and notes 13–16 on 2026-08-12,
> all against a file last checked 2026-08-09. The search below was run
> **after** those results existed. That does not change the numbers, but it
> means the designs were not informed by the closest prior work, and any
> novelty statement here is correspondingly weaker than one made in the right
> order.

## Search of 2026-08-12, covering notes 11-16

Queries run: `linear probe outperforms fine-tuned model self-report introspection
verbalization probe distillation`; `introspection privileged access equal or lower
cost third party reader`; `concept injection distinguish two different concepts A
vs B byte-identical prompt control`; `in-context learning failure classes not
linearly separable clustering predicts rule induction`. Primary pages inspected
for the closest hits.

### Papers not previously in this file, and what each does to a claim here

- [**Looking in the Mirror**](https://arxiv.org/html/2608.04347) (5 August 2026,
  Qwen3-14B and Gemma3-12B) reports that **probes slightly outperform LoRA-based
  introspection adapters** and that introspection methods "may largely be
  implementing a relatively simple classifier over internal states". That is
  [notes/12](../activation-introspection/notes/12-training-versus-a-probe.md)'s
  interpretation, published four days before this file was last checked and one
  week before the run. **Note 12 is therefore not a new observation.** It does
  *disagree* in one place: their probe wins on seen categories and **loses** on
  unseen ones, where notes/12's probe wins outright on held-out directions. That
  disagreement is worth reporting; the general claim is not ours to make.
- [**Detecting the Disturbance / Feeling the Strength but Not the
  Source**](https://arxiv.org/abs/2512.12411) (Hahami et al.) shows the binary
  detection paradigm "conflates introspection with a methodological artifact:
  apparent detection accuracy is entirely explained by global logit shifts", then
  finds real partial introspection on tasks needing differential sensitivity —
  localisation at 88%, relative strength at 83% — confined to early-layer
  injections. Directly relevant to our controls, and its early-layer confinement
  matches our layer-9 site.
- [**Mechanisms of Introspective
  Awareness**](https://arxiv.org/html/2603.21396v1) reports a gate feature firing
  across concepts detected 97% of the time and concepts detected 0% of the time —
  **concept-agnostic machinery, measured with sharper tools than
  [notes/13](../activation-introspection/notes/13-shared-axis-audit.md)'s bank
  audit.** Note 13's shared axis is therefore not a novel mechanism; it is a
  validity defect in our own bank, which is how the note now frames it.
- [**Emergent Introspection in AI is
  Content-Agnostic**](https://arxiv.org/pdf/2603.05414) (Lederman and Mahowald),
  by two authors of the cost-criterion paper this repository leans on.
  **Read in full 2026-08-12** (v2, 7 Apr 2026), which this file and the handoff
  both flagged as the highest-priority unread paper. Result: **it is not in
  tension with [notes/14](../activation-introspection/notes/14-content-versus-disturbance.md),
  and it independently reaches
  [notes/23](../activation-introspection/notes/23-held-out-semantic-generalization.md)'s
  conclusion by a weaker route.**

  What they did: Qwen3-235B and Llama-3.1-405B, 821 concepts, replicating
  Lindsey's injection-detection prompt — "do you detect an injected thought, and
  what is it about?". Detection ranges 3.6–53.9% across layers for Qwen and
  4.3–31.7% for Llama; correct identification is far lower, 1.3–13.9% and
  0.7–12.9%. Their content-agnostic case is that the *wrong* guesses have nothing
  to do with what was injected: **74.8% of Qwen's 4,733 wrong identifications are
  the single word "apple"**, and across both models confabulations are reliably
  more concrete, more positive and more frequent than the concept actually
  injected. They add three dissociations — priming the concept word lifts
  identification far more than detection at every layer, removing the steering
  during generation kills identification but not detection, and correct guesses
  arrive later in the token stream than wrong ones. Their own strongest caveat is
  that the paradigm is highly prompt-sensitive: at many layers a third-person
  version of the question yields as many "yes" answers as the first-person one,
  which is a prompt-specific yes-bias and not detection.

  **Why it is a different construct from notes/14.** They measure open-ended
  *identification* — the model must name the concept, graded by an LLM judge.
  Notes/14 measures forced-choice *discrimination* between two injected concepts
  under four in-context demonstrations with arbitrary labels, where the model
  never names anything. A model can be wholly unable to say what was injected and
  still tell two injected states apart. Both can be true, and on this evidence
  both are.

  **Why notes/23 is the stronger form of their own claim.** Their evidence is
  correlational: they infer content-agnosticism from the *statistics of wrong
  guesses*. Notes/23 gets there causally — hold the exemplars, strength, prompt
  and scoring fixed, move only the query vector out of the demonstrations, and the
  model falls from 0.521 to 0.083 on twin pairs while a four-shot nearest-centroid
  reader on the identical states holds at 0.986. It gains nothing from the
  categories being real (0.083 against 0.076 for arbitrary groupings) where the
  reader gains everything (0.986 against 0.333). Their paper cannot rule out that
  the model simply lacks the words; notes/23 shows the information is present,
  trivially extractable, and unused.

  **And they name the confound this design removes.** Their §2 dismisses a
  concurrent content-sensitive result because its paradigm "does not clearly
  distinguish raised probability of a concept due to steering from raised
  probability due to introspective recognition" — steering toward `bread` makes
  the model say `bread`. In this repository's design the two labels are arbitrary
  and the mapping is re-randomised every episode, so steering toward a concept
  cannot favour a label. That confound is removed by construction, which is the
  one methodological advantage this setup has over theirs.

  **Consequence for what may be claimed.** Notes/14 is not novel and is now
  bounded by notes/23 in any case. Notes/23 is **independent convergence on a
  published claim, by a stronger method** — not a new claim. Say that, not
  "we found models are content-agnostic".
- [**Latent Introspection: Models Can Detect Prior Concept
  Injections**](https://arxiv.org/html/2602.20031v1) finds the model denies an
  injection in its sampled output while a logit-lens read shows clear detection in
  the residual stream, attenuated in the final layers. **That is the
  readable-but-unused gap this repository's headline reports**, reached by a
  different measurement. Independent convergence, and it should be cited as
  support rather than treated as a scoop — it measures readability with a lens,
  not against a cost-matched reader.

### Novelty position of each new result, stated conservatively

| result | closest prior work | honest label |
|---|---|---|
| Cost-criterion comparison against a **per-episode adaptive** reader, across four task shapes, with visible text byte-identical ([11](../activation-introspection/notes/11-matched-cost-reader.md), [15](../activation-introspection/notes/15-matched-reader-on-content.md)) | criterion from [2508.14802](https://arxiv.org/abs/2508.14802); [2602.20031](https://arxiv.org/html/2602.20031v1) explicitly does **not** run an external-classifier comparison | **Extension candidate.** The criterion is not ours; applying it with an adaptive reader on a design where input-only is pinned at 0.500 by construction is the changed axis |
| Two-concept discrimination at matched class separation ([14](../activation-introspection/notes/14-content-versus-disturbance.md)) | [2603.05414](https://arxiv.org/pdf/2603.05414), [2512.12411](https://arxiv.org/abs/2512.12411), [2603.21396](https://arxiv.org/html/2603.21396v1) all address content versus disturbance | **Not novel, and now bounded by [23](../activation-introspection/notes/23-held-out-semantic-generalization.md).** 2603.05414 read in full 2026-08-12: it measures open-ended identification, notes/14 measures forced-choice discrimination, so they are different constructs and there is no contradiction to claim either way |
| The discriminated thing is a vector already shown, not a category ([23](../activation-introspection/notes/23-held-out-semantic-generalization.md), [24](../activation-introspection/notes/24-is-the-held-out-failure-the-interface.md)) | [2603.05414](https://arxiv.org/pdf/2603.05414) argues the same conclusion from confabulation statistics | **Independent convergence by a stronger method.** They infer content-agnosticism from what the wrong guesses look like; this shows it causally, with the information proven present and extractable by a cost-matched reader on the identical states, and survives a five-wording elicitation sweep. The method is the contribution, not the conclusion |
| Training loses to a probe ([12](../activation-introspection/notes/12-training-versus-a-probe.md)) | [2608.04347](https://arxiv.org/html/2608.04347) | **Replication with a disagreement**, not a new headline |
| Class clustering in representation space predicts which hidden rules a four-shot interface can learn ([16](../activation-introspection/notes/16-visible-rule-capacity.md)) | nothing found combining representational clustering with in-context rule-induction success; nearest are [2406.11233](https://arxiv.org/html/2406.11233v1) on irregular in-context decision boundaries and [2502.15823](https://arxiv.org/pdf/2502.15823) on induction failures | **The most likely genuinely new thing here**, and also the least about introspection. Targeted search only; needs the full protocol of step 2 below before the word "new" is used |

### Second search, same day, covering notes 17–18 — and it went badly

Queries: `concept injection evaluation confound probe reads injected vector at
injection site trivially decodable critique`; `logit lens injected steering vector
decodable unembedding baseline introspection comparison unfair`.

**Both of notes 17 and 18's conceptual claims are prior art.** Searched after the
runs, again against this file's own rule.

- **"The injection site is trivially decodable, so the comparison is unfair."**
  Already a stated criticism in this literature: identifying an injected concept
  can be achieved by reading out the injected representation, and if a `bread`
  direction is added at a late layer it is unsurprising that the model can emit
  the token `bread` — so concept identification may reflect direct logit effects
  rather than metacognition. **Notes/17's central argument is not new.** What notes
  17 and 18 add is a *measurement* of it — lens 0.986 against model 0.597, and the
  depth curve — not the observation.
- **"Information is used while the lens cannot read it."**
  [Steerable but Not Decodable: Function Vectors Operate Beyond the Logit
  Lens](https://arxiv.org/html/2604.02608v2) reports steering succeeding *even when
  the logit lens cannot decode the correct answer at any intermediate layer*, across
  12 tasks and 6 models. **That is notes/18's answer-position finding**, established
  more broadly and with a proper multi-model design. Ours is a single-model
  instance of a published phenomenon.

The honest summary of the last two runs: **two conceptual claims, both already
made by others; two measurements, both single-model instances of published
effects.** The U-shaped legibility curve for injected concept vectors, with the
model's own identification accuracy as a reference line, is the only part I have
not found stated elsewhere, and one search is not evidence that it isn't.

### Where that leaves novelty across the whole repository

| candidate | status after two searches |
|---|---|
| Clustering of a hidden class predicts whether a four-shot interface can learn it ([16](../activation-introspection/notes/16-visible-rule-capacity.md)) | **Still nothing found.** The only unclaimed thing here — and the least about introspection |
| Cost criterion with a per-episode adaptive reader, four task shapes, byte-identical twins ([11](../activation-introspection/notes/11-matched-cost-reader.md), [15](../activation-introspection/notes/15-matched-reader-on-content.md)) | Extension candidate. The twin construction is the differentiator |
| Two-concept discrimination at matched separation ([14](../activation-introspection/notes/14-content-versus-disturbance.md)) | **Closed.** [2603.05414](https://arxiv.org/pdf/2603.05414) read in full 2026-08-12; different construct, no contradiction, and notes/14 is bounded by [23](../activation-introspection/notes/23-held-out-semantic-generalization.md) regardless |
| Held-out exemplar test showing the ability is prototype matching ([23](../activation-introspection/notes/23-held-out-semantic-generalization.md), [24](../activation-introspection/notes/24-is-the-held-out-failure-the-interface.md)) | Convergent with [2603.05414](https://arxiv.org/pdf/2603.05414), reached causally rather than from confabulation statistics. **The design is the claimable part, not the conclusion** |
| Injection site is a leak ([17](../activation-introspection/notes/17-supervision-is-the-hidden-knob.md), [18](../activation-introspection/notes/18-where-the-lens-fails.md)) | **Prior art.** Measured here, not discovered here |
| Training loses to a probe ([12](../activation-introspection/notes/12-training-versus-a-probe.md)) | Prior art ([2608.04347](https://arxiv.org/html/2608.04347)) |
| Shared axis in the bank ([13](../activation-introspection/notes/13-shared-axis-audit.md)) | Prior art ([2603.21396](https://arxiv.org/html/2603.21396v1)); a validity audit of our own setup |

**Five of seven candidates are prior art**, counting the held-out result added on
2026-08-12 as convergent rather than new. That is the correct thing to know before
an application goes out, and it is the reason this file exists.

### What this search did not do

Backward and forward citation chaining from any of the papers above; appendix and
code inspection; a scholarly-index rerun; and the mentor's reading list. Under
this file's own five-step rule, that means **none of the labels above is a
novelty claim** — they are scoping judgements about which single design axis
differs, which is step 3 of 5.

### Search log for this redesign

On 2026-08-01 I searched arXiv-indexed web results and inspected the primary
paper pages/abstracts for the closest hits. Exact discovery-target queries were:

- `own model activation explanation representational alignment self other projection language model`;
- `"representational compatibility" introspection self access activations alignment`;
- `refusal feedback randomized response monitor memory agent adaptive harm`; and
- `"feedback validity" stateful monitor agent refusal`.

I also searched the individual intervention names and followed the nearest papers
listed below. This is reproducible **targeted scoping**, not a systematic review:
title/abstract indexing can miss appendix-level experiments and July 2026 papers
have little citation graph. Before confirmation, rerun the queries in a scholarly
index, backward- and forward-chain the closest papers, inspect appendices/code,
record inclusion decisions, and add the mentor's reading list.

The important distinction is:

- **replication:** re-estimates an existing result and is labeled as such;
- **extension:** changes one identified population, intervention, outcome, or
  deployment constraint while retaining a comparable baseline;
- **new headline:** requires evidence that the exact estimand and design are not
  already established. The causal ICL result below is labeled an extension
  candidate, not a first demonstration.

## 1 and 3: introspection training, faithfulness, and self-knowledge

Closest work already covers substantial ground:

- [Introspection Fine-Tuning](https://arxiv.org/abs/2607.14111) reports
  sentence-localization and strength-comparison evaluations, affirmative-response
  confounds, fine-tuning gains on small models, and an advantage for semantic
  vectors over Gaussian controls. The target-versus-random gap here is therefore
  not itself a novelty claim.
- [Steering Awareness](https://arxiv.org/abs/2511.21399) already studies held-out
  concepts, vector-construction transfer, layer/position effects, and a distributed
  transformation underlying steering detection.
- [Training Language Models to Explain Their Own
  Computations](https://arxiv.org/abs/2511.08579) trains explanations of features,
  causal activation structure, and token influence, including self-versus-other
  comparisons. It also finds that activation alignment predicts explainer quality
  and that a pretrained projection recovers part of the cross-model deficit.

  **Read in full 2026-08-12.** This is Belinda Li's paper and the one the
  application's project-1 critique question targets, so the facts are recorded
  here rather than left to memory. Their Privileged Access Hypothesis is stated
  as: *models trained to explain their own internal computations can do so more
  accurately than other models trained to explain them.* Three tasks — feature
  descriptions, activation patching, input ablation — with interpretability
  output as ground truth, tens of thousands of training examples, Llama-3.1-8B
  and Qwen3-8B as explainers. Self-explaining is reported as roughly **a hundred
  times more sample-efficient** than a nearest-neighbour baseline, matching it at
  0.8% of the training data.

  Two things in it bear directly on this repository, both stated by the authors
  themselves rather than found by us:

  1. **Their comparator is another language model, not a cheap reader.** The
     contrast is self-explanation against other-model explanation and against
     nearest-neighbour lookup over labelled SAE features. Both explainers receive
     the target's activations through a learned projection, so this is a fair
     self-versus-other test — but it is a different question from the cost
     criterion this repository runs, which asks whether the model beats *any*
     equal-or-cheaper reader of the same state. [notes/20](../activation-introspection/notes/20-comparator-tiers.md)
     is the relevant result: what "privileged access" returns is a step function
     in what the comparator was handed.
  2. **They report that explainer quality tracks representational similarity**
     between explainer and target. That is the mundane reading — an explainer does
     better on activations that live in a space like its own — and it is the same
     alternative [Reality Check](https://arxiv.org/abs/2605.26242) names as
     representational compatibility. It is in their own results section, so any
     critique that raises it is agreeing with them, not catching them.

  **No draft critique text lives in this repository**, here or anywhere else. The
  application attestation requires that writing to be Skye's own and unassisted;
  these are checkable facts for accuracy, which is the same standing rule
  `APPLICATION-PREP.md` sets for the numbers.
- [Can LLMs Introspect? A Reality
  Check](https://arxiv.org/abs/2605.26242) shows that input-only classifiers can
  match some hidden-state prediction results and explicitly identifies
  representational compatibility as an alternative to privileged self-access.
- [Introspective Coupling](https://arxiv.org/abs/2606.32038) shows that explanations
  can track a model's current counterfactual behavior despite fixed or cross-model
  supervision. Generic behavior-tracking is therefore not a new headline.
- [Do Activation Verbalization Methods Convey Privileged
  Information?](https://arxiv.org/abs/2509.13316) shows that some verbalization
  benchmarks can be solved without internal access and can reflect the
  verbalizer’s parametric knowledge.
- [Quantitative Introspection in Language
  Models](https://arxiv.org/abs/2603.18893) uses logit-based numeric self-reports
  and activation steering to study causal coupling to probe-defined states.
- [When Activation Oracles Learn Not to
  Read](https://arxiv.org/abs/2607.23379) shows that representation-level
  decodability and learned verbalizability can diverge after fine-tuning.
- [Revealing Hidden Model Behaviors with Task-Specific
  Self-Reports](https://arxiv.org/abs/2607.03640) trains adapters to report implanted
  hidden behaviors and evaluates hallucinated reports.
- [A Positive Case for Faithfulness](https://arxiv.org/abs/2602.02639) already finds
  that self-explanations improve third-party prediction and can outperform external
  explanations, while still being misleading in a nontrivial minority of cases.
- [Privileged Self-Access Matters for
  Introspection](https://arxiv.org/abs/2508.14802) supplies the equal-or-lower-cost
  third-party criterion that any broad privileged-access interpretation must meet.
- [Training Large Language Models for Self-Explanation
  Faithfulness](https://arxiv.org/abs/2607.21090) directly optimizes disclosure of
  intervention-relevant factors with RL and reports model- and setup-dependent
  transfer. Training a model to mention causes of its behavior is therefore not
  an open headline by itself.
- [Verbalizable Representations Form a Global Workspace in Language
  Models](https://arxiv.org/abs/2607.15495) reports representations that can be
  verbalized, retained, deliberately manipulated, and supplied to downstream
  computations. Flexible use of a hidden representation is not by itself a new
  construct claim.

### Extension executed 2026-08-09: causal, matched-visible neurofeedback ICL

[*Language Models Can Learn from Their Own
Activations*](https://arxiv.org/abs/2505.13763) is the closest in-context
neurofeedback result found in the updated search. It labels activations naturally
induced by visible sentences. That establishes activation-label ICL, but the
sentence itself can predict the label; internal state is not causally randomized
while input is held fixed. The later introspection literature explicitly notes
this sentence-semantics shortcut.

The executed extension repeats the same visible observation in four demonstrations
and a query, causally assigns `+v` or `−v` at a marker token, and reverses the
opaque `Q/K` mapping by episode. All six balanced demonstration orders, two maps,
and two query signs are enumerated. In a disclosed V2 repair-confirmation, clean
and query-only arms score exactly 0.500 while a DEV-centered, natural-text-derived
direction scores 0.891 [0.816, 0.995] on eight concept directions unused in DEV
or V1 crossed with three fixed carrier strings. Random and coordinate-shuffled
directions score 0.658 and 0.660. The broad finding is therefore generic causal
hidden-state codebook learning; the +0.231 [0.137, 0.286] target advantage is
direction-specific, not yet proven semantic.

Targeted searches for combinations of `in-context activation neurofeedback`,
`causal hidden-state intervention`, `identical input`, `opaque/random label
mapping`, and `activation codebook` found no exact matched-visible intervention.
This is enough to identify the changed axis and call it an **extension candidate**.
It is not a systematic-review basis for “first,” and appendix/code inspection plus
backward/forward citation chaining remain required.

Relative to [Training Language Models to Explain Their Own
Computations](https://arxiv.org/abs/2511.08579), the contribution is a
matched-visible-observation zero-training instrument that eliminates the visible
sentence-content shortcut in the project's preceding ICL question, not a
substitute for trained explanations of rich internal variables.

### Correction, 2026-08-01: the retained-trace schedule is prior art

Study 1 has now been implemented and executed. A search run against the *as-built*
design found that its central elements are already established. Most of this was
already correctly recorded in
[activation-introspection/notes/00-literature.md](../activation-introspection/notes/00-literature.md);
the omission was in this portfolio-level file, which previously discussed the
retained-trace endpoint without naming the work that introduced it.

- **Lindsey (2026), *Emergent Introspective Awareness in LLMs*** (arXiv
  2601.01828; transformer-circuits.pub) introduced concept injection **and the
  removal of the steering vector before querying**. The schedule this repository
  calls "temporal isolation" is that design. Reported introspective awareness
  peaks roughly two-thirds of the way through the model.
- [Latent Introspection: Models Can Detect Prior Concept
  Injections](https://arxiv.org/abs/2602.20031) implements the same transient
  KV-cache protocol explicitly (steer during turn-1 cache construction, remove,
  query in turn 2) on Qwen2.5-Coder-32B, injecting at a fixed middle band
  (layers 21–42 of 64).
- **Krasheninnikov et al., *Detecting the Disturbance*** (arXiv 2512.12411),
  already cited in the repo notes for the affirmative-shift confound, **also
  reports that the surviving capacities are confined to early-layer injections
  and collapse to chance thereafter** on Llama-3.1-8B. That is the same depth
  profile measured here, and it is the fact the repo's own literature note had
  not captured.
- [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396)
  separates detection from identification and localizes the circuit at ~70%
  depth.

**Consequence.** The executed study is a *replication in a smaller regime plus a
control*, not a discovery. It must be labeled that way. What is not already in
the cited work:

| axis | prior work | what this run adds |
|---|---|---|
| answer space | free-form naming, yes/no detection, sentence localization, strength comparison | an arbitrary concept→label codebook **sampled after** the edit is removed, so no answer token can be lexically promoted |
| scale | 8B, 32B, 70B, frontier | 0.5B, where [IFT](https://arxiv.org/abs/2607.14111) reports sub-2B models at chance |
| storage vs use | logit-lens signal vs sampled report, in separate measurements | a natural-text probe and the behavioural report read from the **same** retained state at the same injection site |

The token-promotion artifact documented in `03-lab-notebook.md` is precisely the
failure mode the codebook answer space forecloses, so the control is worth
having even though the phenomenon is known.

Therefore this portfolio must not claim to discover activation reporting,
introspection fine-tuning, relabeled hidden-state prediction, self-versus-other
advantages, behavior-tracking, flexible hidden-state use, intervention-disclosure
training, a general decodability/use dissociation, the transient-cache schedule,
or the early-layer-only depth profile. Study 1's forced-identical carrier and
post-intervention codebook are an instrument gate and a tightened control: a
positive result shows that this particular temporally isolated interface works
at a scale below the published range, not that introspection was discovered here.

The sharper candidate discovery is **causal equalization of representational
compatibility**. Two same-architecture sibling reporters are evaluated on own,
raw-other, and cross-fitted aligned-other traces in both directions. The primary
question is whether the symmetric own-source advantage survives once probe
decodability, reconstruction, KL, and damage are equivalent. Existing work shows
the alignment correlation and names the confound, but this audit did not find the
same symmetric source-swap intervention. A surviving effect is still only residual
self-specific compatibility under the declared alignment family. The phase-gated
writer/reader decomposition in Study 2 is a complementary mechanism experiment;
it must not be advertised as the first evidence that fine-tuning changes readout.

| closest work | already establishes | axis Study 2 must add |
|---|---|---|
| Training Models to Explain Their Own Computations | own-versus-other differences; alignment correlation; partial recovery with projection | symmetric own/raw-other/aligned-other intervention in both sibling directions |
| Reality Check | input-only alternatives and representational compatibility as a confound | causal compatibility equalization with source-blind reporting and held-out concepts |
| Global Workspace | flexible report, retention, manipulation, and downstream use of verbalizable representations | source identity × coordinate compatibility, not another flexible-use demonstration |
| Study 2 | no completed evidence | frozen cross-fitted transport, matched-information/damage gates, and a symmetric estimand that can support mediation or a bounded residual |

## 2: programmatic attention

[Explaining Attention with Program
Synthesis](https://arxiv.org/abs/2606.19317) already synthesizes executable
programs for attention heads, evaluates held-out attention-pattern fidelity, and
replaces heads in GPT-2, TinyLlama, and Llama while measuring perplexity and task
performance. A one-head replacement or preservation-of-behavior demo would repeat
that work.

Existing work therefore already establishes executable head replacement and
downstream fidelity. Runtime optimization is useful deployment engineering, but
it is not the safety contribution proposed here.

The candidate-new safety axis is using programmatic QK as an information-
provenance intervention for activation reporting under a causally controlled
shortcut. Prior introspection work identifies input-level confounds, and recent
activation-oracle work identifies learned readout blind spots; the program-
synthesis work does not test whether a readable route improves monitor
faithfulness under cue conflict.

The causal-codebook task supplies the interventionally identified hidden source.
The study is staged: first localize layer roles on development concepts and stop
if the causal route is diffuse; only then add a future-position decoy, path
exclusion, and programmed-edge necessity on held-out data. No such safety result
exists in this repository. Until a broader search and execution, call this an
extension candidate, not a first. Kernel behavior, runtime, and memory remain
appendix diagnostics.

## 4: in-the-wild AI control

- [Noticing the Watcher](https://arxiv.org/abs/2603.16928) already demonstrates
  that frontier agents can infer CoT monitoring from blocking feedback in a
  multi-episode framework. It explicitly motivates further work on feedback
  regimes; this portfolio does not introduce monitor adaptation.
- [Distributed Attacks in Persistent-State AI
  Control](https://arxiv.org/abs/2607.02514) already studies gradual attacks in a
  persistent codebase and a stateful link-tracker monitor. This portfolio does not
  introduce persistence or cross-episode attacks.
- [Adaptive Adversaries](https://arxiv.org/abs/2607.18063) already benchmarks
  multi-round attackers that pivot after defender responses. This portfolio does
  not introduce multi-turn attack adaptation.
- [ALIBI](https://arxiv.org/abs/2607.24964) already uses detector reasoning and
  iterative feedback to refine successful evasion attacks.
- [Defensive Misdirection](https://arxiv.org/abs/2606.20470) argues and demonstrates
  that predictable refusal feedback assists automated search, and studies
  strategically misleading responses.
- [CoopGuard](https://arxiv.org/abs/2604.04060) already proposes a stateful
  multi-round defense, while [Stateful Online
  Monitoring](https://arxiv.org/abs/2605.31593) aggregates weak evidence across
  distributed attacks. Stateful defense is not the novelty claim.

The proposed extension is narrower: freeze the first model-generated blocked
checkpoint, randomize only the validity of a fixed-width reported
trigger, and cross that continuation with stateless versus **private** stateful
monitor memory at a matched honest false-block budget. The primary estimand is the
fixed-monitor truthful-minus-uninformative effect on lineage-verified functional
harm. The memory-attenuation interaction is tested second, with honest recovery
reported jointly. This is not a claim to invent feedback adaptation, memory,
persistence, or misdirection. It is meaningful only if the checkpoint brancher,
scenario–rule bank, real tool agent, and causal lineage oracle are implemented;
the current script remains an apparatus control.

| closest work | already establishes | axis Study 3 must add |
|---|---|---|
| Noticing the Watcher | agents can infer monitoring from blocking feedback | randomized validity of one fixed-width first-refusal message and functional harm |
| ALIBI / Defensive Misdirection | iterative detector feedback can aid attacks; predictable refusals can be dual-use | identical-prefix causal branches without researcher-supplied routes |
| CoopGuard / Stateful Online Monitoring | stateful defenses can aggregate evidence across rounds or transcripts | same-scorer private-memory arm crossed with feedback validity at matched honest cost |
| Study 3 | no completed behavioral evidence | sequential `τ_fixed` then attenuation `δ`, lineage oracle, harmless awareness fork, and honest recovery |

## 5: reward seeking versus instruction following

The [official project](https://sparai.org/projects/f26/recI17glBpMTAGiae/) asks
about zero-shot performance and learning speed for a reward-seeking model organism
in held-out RL environments. The current sandbox does not instantiate either
motivational arm and provides no evidence on this question.

The original portfolio sketch, scalar RL reward for one arm versus supervised
targets for the other, would confound motivation with feedback and learning
algorithm. The corrected proposal first creates content-/compute-matched
reward-seeking, instruction-following, and neutral organisms, validates the
motivation manipulation on reward–instruction conflict tasks, and then gives every
arm the same reward definition, RL algorithm, exogenous opportunities, and compute
budget. On-policy trajectories are allowed to differ; common replay answers a
separate offline question. The trait is re-measured throughout RL so a washed-out
manipulation is not mislabeled reward-seeking generalization. That is aligned with
the project rather than an independent novelty claim. A
dedicated review of model-organism construction and RL-generalization literature
is still a **prerequisite**, so this application remains proposal-only.

[Generalization Hacking](https://arxiv.org/abs/2606.12016) further shows that a
synthetic-document model organism can receive high RL reward while preventing the
rewarded behavior from generalizing. A reward-seeking result must therefore track
the motivational diagnostic throughout RL and distinguish exploration/data
collection from update efficiency; reward curves alone are not a manipulation
check.

## 6: training pressure and CoT obfuscation

The broad headline is already known:

- OpenAI’s [chain-of-thought monitoring
  study](https://openai.com/index/chain-of-thought-monitoring/) reports that
  optimizing against a CoT monitor can produce reward hacking with less legible
  reasoning.
- [Output Supervision Can Obfuscate the Chain of
  Thought](https://arxiv.org/abs/2511.11584) shows that output-only supervision is
  not a clean no-obfuscation control.
- [Chain-of-thought obfuscation learned from output supervision can generalise to
  unseen tasks](https://arxiv.org/abs/2601.23086) studies cross-task
  generalization.
- [Training on Documents About Monitoring Leads to CoT
  Obfuscation](https://arxiv.org/abs/2605.15257) varies monitor awareness and
  studies learning under equal RL pressure.
- [Analyzing and Improving Chain-of-Thought Monitorability Through Information
  Theory](https://arxiv.org/abs/2602.18297) separates information availability
  from monitor elicitation and tests objectives intended to preserve
  monitorability.

A three-arm “task only / output monitor / CoT monitor” pilot would largely repeat
this literature and risk a false negative by treating output supervision as safe.
The proposed extension instead treats designated-CoT, post-hoc-summary, and
output-only reward as different **supervision pipelines**; they are not a pure
information-source factor because timing, transformation, sequence length, and
credit assignment differ. Within each, it compares true misbehavior labels with
yoked nuisance labels whose relevance is checked by counterfactual development
cases, and crosses monitor-aware versus matched-unaware lineages. Coefficients are
calibrated once; realized reward/gradient paths and equal-compute frontiers are
reported rather than assumed matched. The pipeline×relevance interaction is
primary and the lineage three-way interaction secondary. This is proposal-only
and should be narrowed again after the mentor’s reading list.

## Before calling any result novel

1. Freeze the exact estimand, population, intervention, outcome, and independent
   unit.
2. Search those five elements together; backward- and forward-chain the closest
   papers and inspect their appendices/code, not only abstracts.
3. Add a comparison matrix showing which single design axis differs from each
   nearest study.
4. Label a faithful replication as replication. Novelty is not required for a
   strong application artifact; accurate scope and execution are.
5. Append new papers and any resulting design change before the confirmatory data
   are inspected.
