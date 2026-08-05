# Claim audit and v2 preregistration

Date: 2026-08-01. This document supersedes the conclusions, not the historical
record, in `03-lab-notebook.md`. It is written before rerunning the repaired
pipeline. Existing result files are preserved as exploratory artifacts.

> **Update.** The retained-trace endpoint proposed below as "proposed and not
> implemented" has since been built and run. Its preregistration, results, and
> prior-art check are in [`05-retained-trace.md`](05-retained-trace.md). The
> executive decision below still stands as written: that study is a replication
> of a published effect at a smaller scale, so it is not a *new* scientific
> result. Everything else in this file is unchanged.

## Executive decision

The repo currently demonstrates useful engineering and error-finding, but it
does not yet establish a new scientific result. The old central claim was a
false positive caused by comparing different intervention sites. The old null
claim has material false-negative risk from model scale, answer-format tax,
observer asymmetry, intervention damage, and an invalid independence assumption.

## Claim ledger

| ID | claim previously made | status | evidence and decision |
|---|---|---|---|
| A1 | Fixed-source probe transfer anti-correlates with post-IFT accuracy, r = −0.774. | **Retracted** | Probe transfer injected at L8 and read at L; IFT injected at L and read at output. Opposite depth trends are built into the estimands. `run_ift.py` now rejects that profile. |
| A2 | “Decodability by a probe is not usability by the forward pass” is demonstrated here. | **Retracted** | A local matched-site reconstruction instead gave positive descriptive correlations: L8-trained .867, L16-trained .973, pooled rows .905, per-layer means .940. The aggregate lacks raw trials and provenance, so even the corrected association is exploratory. General decodability/causal-use dissociations are already established in prior work. |
| A3 | A minimum remaining-compute budget governs IFT generalization. | **Hypothesis, not mechanism** | The edit is a post-block hook. At L23 it occurs after the final LoRA-bearing block, so failure is structurally expected. L1 is a counterexample to sufficiency: 22 blocks remain, yet legacy post-IFT accuracy was .175. Test pre- vs post-block placement and a trainable downstream readout. |
| A4 | Training at mid-depth is generally optimal. | **Unsupported** | Two adapter runs on one 0.5B model, one strength, one task, and invalid legacy evaluation cannot support a prescription. |
| A5 | The negative introspector−observer gap establishes no privileged access at 0.5B. | **Unsupported** | The old interval treated repeated cells as IID. A concept×prompt cluster bootstrap gave [−.236, .047]. The observer also has clean activations and a different transcript, creating a structural advantage. Defensible wording: no positive evidence under this instrument. |
| A6 | Word-scored identification while the edit is live measures introspection. | **Falsified locally** | A no-question control reproduced 1.000 accuracy. This identifies token promotion in this schedule. It does not invalidate removal-before-query designs. |
| A7 | The full-depth fixed-source profile has a structural high band at 58–75% depth. | **Exploratory** | One model, one source layer, one strength. Near-source behavior changed when the source moved by one layer; near-output behavior is entangled with token promotion. |
| A8 | The IFT and Introspection Adapters papers explicitly state the quoted probe/generalization gaps. | **Withdrawn attribution** | The audit could not verify the exact quotations. They were at most an inference by the repo author and should not be presented as paper text. |

## Code-level invalidations found in the audit

1. LoRA evaluation ran in training mode, leaving adapter dropout active. The
   apparent variation across repeated prompt ids was partly dropout noise.
2. Three training and two evaluation paraphrases were repeated many times and
   treated as seeds. They are not independent experimental units.
3. The option list was fixed, so every concept had a permanent digit label. An
   adapter could memorize concept→digit rather than read a trial-specific menu.
4. The vector bank built at the training layer was injected at every held-out
   layer. The repaired runner constructs and validates a bank at each site.
5. `layer_profile.py` and `probe.collect` applied the intervention twice to the
   self-report forward: an outer hook remained live while `score_choices`
   registered the same edit again.
6. The old observer interval resampled dependent trials as IID. The prompt and
   concept clusters, and adapter-training runs for IFT, are the relevant units.

## Threats in both directions

### Ways to manufacture a positive

- score the same lexical direction that was injected;
- fix the concept→answer mapping across training and evaluation;
- compare a fixed-source propagation curve with an inject-at-site curve;
- train and test a high-dimensional probe on overlapping templates or injected
  examples;
- tune layers, strengths, prompts, and metrics on the same test set;
- leave stochastic dropout active and count repeated masks as independent runs;
- apply an intervention twice in one arm;
- report a selected best layer without multiplicity correction; or
- treat an arbitrary validity gate as though it guarantees construct validity.

### Ways to erase a real effect

- use a model below the capability scale at which the behavior emerges;
- require concept→digit indirection without a matched format-comprehension
  positive control;
- inject strongly enough to damage the introspector while giving the observer a
  clean forward pass;
- inject after the last trainable or causally useful block;
- train a low-capacity adapter or too little data and call non-learning a null;
- use concept vectors that do not reach the answer state;
- ask only two nearly identical held-out prompts;
- give the observer a richer, cleaner transcript than the introspector;
- use a wide uncertainty interval and interpret “contains zero” as equivalence;
  or
- pool model families when an effect may be architecture-specific.

## V2 preregistration draft

This is a protocol to run next, not a description of completed evidence. Freeze
the configuration and analysis commit before generating confirmatory data.

### Questions and estimands

1. **Reach:** for an injection constructed and applied at L, how often does a
   natural-text-trained readout at the output classify the concept correctly?
2. **Trainability:** after IFT at one site, what is held-out-concept,
   held-out-prompt accuracy at each separately constructed injection site?
3. **Placement:** at equal behavioural KL, how do pre-block, post-block, and
   post-final-block interventions differ? This tests, but does not by itself
   identify, remaining compute versus hook placement and downstream trainability.
4. **Retained-trace use:** after a transient edit is removed, does the carrier
   cache support a later arbitrary-label mapping beyond a frozen sham-hook and
   damage-yoked random/shuffled controls?
5. **Introspection specificity:** if retained-trace use exists, does explicit
   self-report add anything beyond a generic state-conditioned classification
   task using the same trace?

### Proposed retained-trace endpoint: transient carrier then blind codebook

Target naming while a concept vector remains live cannot identify privileged
access: the edit can directly promote target-related answer features. Randomized
digits remove lexical overlap, but a trained concept→digit pathway can still be
driven while the intervention is active. A stronger prerequisite is therefore
a two-stage cached-state design, which is **proposed and not implemented**:

1. Build a short carrier prefix and its KV cache while injecting the concept at
   the preregistered site. No concept name, answer label, or codebook is visible
   during this stage, and no model-generated behavioural transcript is exposed.
2. Remove the hook completely. Only then reveal a fresh random mapping from all
   candidate concepts to opaque labels, append it after the cached carrier, and
   score the label. The mapping is sampled after cache construction, so the live
   edit cannot directly promote the target label.
3. Recompute a clean carrier cache from the identical visible prefix, then reveal
   the identical codebook and query. This clean-cache result is a leakage/chance
   gate, not a difference-in-differences term. Across balanced targets and
   codebooks it must remain within the preregistered chance bound. Exact agreement
   between duplicated clean computations is only an implementation check.
4. On the development split, calibrate random and shuffled interventions to
   match carrier KL, format competence, and unrelated-task degradation without
   semantic alignment; freeze their strengths before the held-out run. Treat a
   wrong-concept edit as a semantic-switch arm, not a damage null: score both the
   opaque label for the concept actually injected and the original target label,
   and report the confusion matrix.

This endpoint has two required positive controls. First, a naturally presented
concept must survive the same carrier delay and arbitrary codebook mapping at
high accuracy; otherwise the model may simply lack the binding/format ability.
Second, a matched readout must verify that the transient injected signal persists
in the cached carrier after hook removal. Failure of either control makes a null
uninformative. The current live-injection report remains a steering/trainability
pilot. Even a clean positive establishes causal use of a retained KV-cache trace,
not a self-model or introspection, until the self-report-specific control passes.

### Factors

| factor | preregistered levels |
|---|---|
| architecture/scale | at least two model families, with two sizes in one family; report each separately before pooling |
| adapter training seed | five independent initializations and data orders per model/configuration |
| concept split | disjoint train, validation, and confirmatory held-out concepts; no prompt contains a test concept during adapter training |
| prompt split | disjoint semantic prompt families, not repeated paraphrase indices |
| option mapping | new deterministic balanced permutation on every example; each concept appears equally often in every slot within a block |
| intervention schedule | transient carrier injection removed before a freshly sampled opaque codebook/report stage; continuous live injection retained only as a secondary steering pilot |
| injection site | preregistered early, middle, late, penultimate, and final sites; both pre-block and post-block hooks |
| strength | per-site values calibrated on a development split to fixed KL bands; calibration frozen before test |
| intervention | target concept; sham hook/no-op edit; coordinate-shuffled norm match; random norm match; wrong-concept semantic switch; and no-edit clean-cache leakage gate |
| reporting task | explicit self-report and generic state-conditioned classification using the same trace and opaque mapping |

### Positive and negative controls

- **Format competence:** state the correct concept in ordinary text and ask for
  the randomized digit. If this is below 90%, the digit endpoint for that model
  is invalid rather than negative.
- **Delayed codebook competence:** cache a naturally stated concept, remove it
  from the visible report query, then reveal the same post-cache opaque codebook.
  The retained-trace endpoint is invalid if the model cannot perform this
  binding and retention task.
- **Injection reach:** the matched runner must beat a retrained permuted-label
  readout at the output. If it does not, a self-report null is not informative.
- **Training competence:** the adapter must learn a non-introspective randomized
  menu task with the same output format and data budget.
- **Token promotion:** score concept words after a neutral prompt with no
  identification question. Word-report results are invalid when this control
  matches the question condition.
- **Damage:** record neutral-task KL, perplexity/format accuracy, and output
  coherence at every site/strength. Compare arms only within frozen KL bands.
- **Leakage:** train natural probes only on natural text, group folds by sentence
  template, and keep injected trials entirely out of readout training.
- **Hook semantics:** unit tests must show one intervention per forward and must
  distinguish pre- from post-block capture.

### Independent units and analysis

- Option permutations are nuisance marginalization, not seeds.
- For IFT, the primary independent unit is an adapter-training run crossed with
  held-out concept and prompt family. For untrained reporting, it is model
  checkpoint × concept × prompt family.
- Fit a preregistered hierarchical logistic model with fixed effects for
  injection site/placement/strength and random intercepts (and, if supported,
  slopes) for checkpoint, adapter seed, concept, and prompt family.
- Report cluster bootstrap intervals that resample adapter runs, concepts, and
  prompt families. Never bootstrap repeated option orders as if IID.
- The primary endpoint is the paired post-cache opaque-label probability-point
  contrast between the target trace and the frozen sham-hook plus each
  damage-yoked random and shuffled control after transient injection removal.
  Report the three contrasts separately as well as their preregistered mean.
  Wrong-concept is a semantic-switch/specificity outcome and does not enter that
  control mean. Clean-cache performance is a leakage gate.
  Randomized-index accuracy under a live edit is a secondary trainability
  endpoint. Word and free-form scores are secondary and gated by token-promotion
  controls.
- The reach/post-IFT layer correlation is descriptive because adjacent layers
  are ordered and dependent. The confirmatory test is the preregistered
  hierarchical site/placement coefficient, not a Pearson p-value over layers.
- Use 5 percentage points as a provisional retained-trace smallest effect of
  interest, justified or revised by a blinded development-set power simulation
  before the locked run. Claim practical equivalence only if the entire cluster
  interval lies inside [−.05, .05]. “The interval overlaps zero” is not evidence
  of absence.
- Control the false-discovery rate across secondary layers/strengths and publish
  every preregistered cell, including invalid and failed controls.

### Stop/go gates

1. Freeze data splits, prompt families, KL calibration, code commit, and analysis.
2. Run unit/smoke controls without inspecting confirmatory labels.
3. Stop a model endpoint as **uninformative**, not negative, if format,
   injection-reach, damage, or training-competence controls fail.
4. Run the confirmatory set once. Any post-hoc repair creates a new version and
   a new held-out set.

## Discovery follow-up: alignment-equalized own-source advantage

The retained-trace endpoint above is an instrument gate, not the intended
scientific headline. If it passes, the next study tests a named alternative to
privileged self-access: an apparent self advantage may arise because a reporter's
weights are geometrically compatible with its own residual coordinates.

Create two same-architecture siblings, `A` and `B`, from one base checkpoint with
independent, compute-matched task/report adapters. For each reporter compare:

1. its own source activation;
2. the raw sibling activation; and
3. the sibling activation mapped into the reporter's space.

Fit `T_B→A` and `T_A→B` with cross-fitted orthogonal Procrustes on disjoint,
unlabeled neutral text. Freeze layer, token site, centering, fit split, and map
before held-out concepts are exposed. A preregistered ridge map is a sensitivity
analysis; selecting a transform after seeing which one erases the gap creates a
new study. Train reporters on a balanced source-blind mixture with every
source/transport condition equally represented, and keep visible carrier,
post-capture codebook, query, options, and answer tokens identical across source
arms.

Let `Δ_R←S` be target correct-label probability minus the prespecified sham-hook
and damage-matched random/shuffled controls. The primary estimand is:

`θ = 0.5 × [(Δ_A←A − Δ_A←T(B)) + (Δ_B←B − Δ_B←T(A))]`

Report both directions and the raw-other gap. A pooled average cannot rescue a
one-direction failure. Require equivalence gates for probe decodability,
reconstruction, norm/covariance, carrier KL, unrelated-task damage, format
competence, and report latency. Add input-only, no-activation, clean-cache,
wrong-concept, random-orthogonal transport, identity/self-transport with matched
transformation error, cycle-consistency, and permuted source-pairing controls.
Every control passes through the same source-specific centering and transport
pipeline as its target arm.
Use at least three independent reporter-training runs; option orders and prompt
surfaces remain nested nuisance variation.

Interpretation is deliberately asymmetric:

- a raw own advantage plus aligned equivalence in both directions supports
  mediation by the tested representational compatibility;
- a residual above the SESOI in both directions supports only self-specific
  compatibility beyond the tested linear map, not metacognition or consciousness;
- surviving probe information with failed verbalization localizes the deficit to
  nonlinear/readout compatibility; and
- a one-direction effect is model heterogeneity and must not be pooled.

Start with one feasible 1–1.5B sibling pair and a blinded power simulation.
Replicate a decisive result on a second architecture or larger checkpoint before
making a population claim. Do not spend the first application-scale budget on the
full layer/scale factorial above: first validate the interface, then estimate this
single symmetric effect.

## Artifact contract

Every new run must include:

- one raw JSONL row per trial;
- model identifier and immutable revision where available;
- git commit, package versions, device/dtype, prompt/template ids, concept split,
  option order, adapter seed, and hook placement;
- checksums linking raw trials to summaries;
- an explicit estimand string so incompatible profiles cannot be joined; and
- a machine-readable validity status and reason for every excluded cell.

`scripts/run_reach_output.py` implements this contract for the matched-site
descriptive reach profile. The IFT runner still needs raw per-trial output and
multi-seed orchestration before it satisfies the full confirmatory contract.

## Literature corrections that constrain novelty

- [Steering Awareness](https://arxiv.org/abs/2511.21399) already fine-tunes
  steering detection/identification and tests held-out concepts.
- [Training Language Models to Explain Their Own
  Computations](https://arxiv.org/abs/2511.08579) already includes self-vs-other
  privileged-access comparisons; retained-trace use alone is not such a
  comparison.
- [Do Activation Verbalization Methods Convey Privileged
  Information?](https://arxiv.org/abs/2509.13316) makes input-only and
  parametric-knowledge controls mandatory.
- [Quantitative Introspection](https://arxiv.org/abs/2603.18893) provides a
  continuous-logit alternative to brittle generated reports.
- [Dissociating Decodability and Causal
  Use](https://arxiv.org/abs/2604.22128) already establishes that general
  distinction.
- [When Activation Oracles Learn Not to
  Read](https://arxiv.org/abs/2607.23379) directly separates decodability,
  leakage, and verbalizability.
- [Task-Specific Self-Reports](https://arxiv.org/abs/2607.03640) and
  [Introspection Adapters](https://arxiv.org/abs/2604.16812) are required
  baselines for hidden-behaviour reporting adapters.
- [Can LLMs Introspect? A Reality
  Check](https://arxiv.org/abs/2605.26242) makes representational compatibility a
  primary alternative to privileged access.
- [Verbalizable Representations Form a Global Workspace in Language
  Models](https://arxiv.org/abs/2607.15495) already reports flexible downstream
  use of verbalizable representations; retained-trace use is not a broad novelty
  claim.
- [Training Large Language Models for Self-Explanation
  Faithfulness](https://arxiv.org/abs/2607.21090) already optimizes disclosure of
  intervention-relevant causes.

The publishable candidate is therefore not “first to train introspection,” “first
to separate decodability from use,” or “first to use a hidden trace.” It is the
symmetric causal test of whether representational alignment removes an apparent
own-source advantage. The retained-trace study is the instrument needed to make
that test interpretable.
