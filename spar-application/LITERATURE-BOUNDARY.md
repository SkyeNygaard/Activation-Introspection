# Literature and novelty boundary

Last checked: **2026-08-09**. This is a targeted audit of the closest primary
papers and the six official project descriptions, not a systematic review. A
fresh search, citation chase, and mentor-provided reading list are required before
locking any claim of novelty.

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
