# Six project briefs

Each brief answers the same six questions: what I would study, what evidence I
currently have, what is missing, the smallest authentic portfolio experiment,
the controls/inference/reproducibility package, and the condition under which I
would stop or proceed.

## 1. Introspection Training for Verbalization Activations

Official project: [Belinda Li — Anthropic](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz)

**Question.** Can training make a model’s verbalizations track causally introduced
activation content on held-out concepts, rather than visible prompt cues or generic
perturbation artifacts?

**Current evidence.** [activation-introspection](../activation-introspection/)
contains the relevant PyTorch intervention and fine-tuning plumbing, plus **one
executed study** with a result that speaks directly to this project. On
Qwen2.5-0.5B, a concept injected into a neutral carrier's KV cache and then
removed is still usable when the injection is early (0.500 against 0.125 chance
at layer 2), and at chance by layer 14. But a probe trained only on ordinary text
recovers that concept from the retained state at **0.958–1.000 for every
injection depth**.

That is the useful part for a training project: **the information is already
stored perfectly; what is missing is readout.** Training aimed at verbalization
would not need to make the model retain more — it would need to make a
late-injected trace reachable. The deficit is also site-specific rather than
uniform, so a single "train it to introspect" objective applied at one layer
would be measuring something different depending on where the content entered.

Second, methodological: the original `r = -0.774` headline used mismatched
injection sites and was retracted, and the executed study is the correctly
site-matched version of that same comparison.

**Gap.** The executed study is a replication — the schedule is Lindsey's and the
early-only depth profile is already published for Llama-3.1-8B. It also does not
isolate privileged verbalization, does not involve any training, and covers one
model family at 0.5B–3B. Nothing here separates "the model reports on its state"
from "a generic classifier over the same state would do as well".

**Smallest authentic experiment.** [Study 1](EXPERIMENTS.md#study-1--causal-use-of-a-retained-activation-trace)
is done and passed, so the interface is validated. Next is the symmetric
[aligned-source experiment](EXPERIMENTS.md#study-2--does-an-own-model-advantage-survive-representational-alignment)
with two independently adapted siblings, asking whether the apparent benefit of
reporting on one's *own* activations survives once the coordinate mismatch is
removed. The executed result constrains its design: it must run at an injection
site where the reporting channel is still alive, or it will just re-measure the
readout collapse.

**Controls, inference, reproducibility.** Transient intervention removed before a
fresh arbitrary codebook is revealed; carrier-only cache and forced identical
visible tokens; clean-cache leakage gate; frozen sham-hook plus
development-calibrated damage-yoked random/shuffled controls; wrong-concept
semantic-switch matrix scored against both actual and intended concepts;
no-question token-promotion and format-comprehension gates; exact order
enumeration; crossed concept/prompt-family inference; equivalence test for a null;
saved logits, cache telemetry, splits, adapter, model revision, and generated
figures. The source experiment adds source-blind reporter training, both transfer
directions, cross-fitted alignment on disjoint neutral text, and equivalence gates
for decodability, reconstruction, KL, damage, and format competence.

**Stop/go.** Stop if visible leakage exceeds chance, damage controls match the
target, or held-out transfer fails. Call a positive cache result only causal use of
a retained trace. If a raw own-source gap disappears after alignment, interpret it
as compatibility mediation. If it survives in both directions, call it residual
self-specific compatibility under the tested transform—not metacognition.

**Fit: strong engineering/audit fit.** This is the portfolio’s closest match, but
the proposed v2 result does not exist yet.

## 2. Deploying Programmatic Attention in Real Transformers

Official project: [Belinda Li — Anthropic](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L)

**Question.** Can a human-readable attention program replace a real head while
preserving its target behavior and offering a measurable inference or training
efficiency trade-off?

**Current evidence.** The activation repository demonstrates hook-level transformer
modification. It contains no program synthesis, QK/OV characterization,
attention-kernel work, or efficiency result.

**Gap.** Topical familiarity with residual streams is not evidence of the project’s
core capability: deploying programs inside attention and benchmarking fidelity,
speed, memory, and trainability.

**Literature boundary.** *Explaining Attention with Program Synthesis* already
generates executable attention programs, replaces heads in GPT-2/TinyLlama/Llama,
and measures perplexity and downstream behavior. Replacing one head and showing
that the model still works would be a replication, not a new deployment result.

**Smallest non-duplicative experiment.** Run an inference-only deployment
crossover first. Select the head and program on a development corpus, then freeze
them and implement the program as a genuinely sparse QK path that never
materializes the dense attention matrix. On locked domains and sequence lengths,
compare native fused dense attention, a dense implementation of the same program,
true sparse execution, head ablation, and a complexity-matched random program.
The discovery target is the **break-even surface** over length, sparsity, fidelity,
latency, and peak memory—not merely a successful replacement screenshot. A later
training-path study is separate and only justified if inference deployment works.

**Controls, inference, reproducibility.** Matched inputs and batch shapes; warm-up
and repeated timing; synchronized device measurements; native-head and head-ablation
baselines; random and complexity-matched programs; held-out sequence lengths;
accuracy, output divergence, end-to-end and kernel latency, peak memory, and
training-step cost. Detect graph breaks and count dense fallbacks. Save the program,
patch, profiler traces, hardware/software manifest, and raw benchmark runs.

**Stop/go.** Stop the “deployment” claim if the program is only an offline
description or if benchmarking is not synchronized and reproducible. Proceed if it
runs in the forward pass and preserves a preregistered fraction of behavior with a
clear, honestly measured trade-off.

**Fit: proposal-only.** This could become a credible bridge, but the
present portfolio should not imply it is already done.

## 3. Faithfulness, Self-Knowledge, and Introspection

Official project: [Noah Siegel — Google DeepMind](https://www.sparai.org/projects/f26/rec3KQAI0JcxJJAce)

**Question.** Can we first establish controlled use of hidden state, then determine
whether a model’s self-report reflects privileged self-knowledge rather than
generic state-conditioned computation, surface leakage, or perturbation damage?

**Current evidence.** The executed retained-trace study gives a concrete
decodable-but-unusable case, which is the shape of the faithfulness problem in
miniature. A linear probe reads the concept off the model's retained state at
0.958–1.000 regardless of injection depth; past mid-network the model itself
answers at chance. Whatever the model would say about its state at those sites,
it is not tracking information that is demonstrably present.

Two design choices make this more than an anecdote. The answer is an arbitrary
letter from a codebook that did not exist when the edit was live, so the
intervention cannot have promoted the correct answer token — which is exactly the
artifact that produced a fake 100% result earlier in this repo. And the two
do-nothing arms land on exactly chance at every site, so the visible prompt
provably carries nothing.

The pilot also exposed why this distinction is hard in the first place: a probe
measured at one causal site was compared against behavior at another, and the
resulting headline had to be retracted.

**Gap.** Nothing here separates introspective self-report from generic
state-conditioned computation. A classifier handed the same retained state would
plausibly do as well, and that control has not been run. The study is also a
replication of a published effect, and covers one model family.

**Smallest authentic experiment.** [Study 1](EXPERIMENTS.md#study-1--causal-use-of-a-retained-activation-trace)
is done and serves as the reporting-interface gate. [Study 2](EXPERIMENTS.md#study-2--does-an-own-model-advantage-survive-representational-alignment)
is the scientific test. Two same-architecture sibling reporters receive own, raw
other, and cross-fitted aligned-other traces in both directions. This directly
tests representational compatibility as the explanation for an own-source
advantage.

**Controls, inference, reproducibility.** Clean-cache leakage gate, frozen
sham-hook plus damage-yoked random/shuffled perturbations, arbitrary labels,
wrong-concept confusion matrix, no-question token-promotion gate, affirmative-bias
and intervention-detection controls, blinded held-out concept bank, crossed
concept/prompt-family inference, free-form output alongside forced-choice per-label
probabilities, with raw logits and log-probabilities retained, and raw state/report
records with prompt and codebook hashes. Study 2 additionally requires symmetric
source swaps, source-blind training, a frozen cross-fitted transform, permuted-pair
and random-transport controls, and matched computational access.

**Stop/go.** Stop the self-knowledge interpretation if the result is explained by
visible leakage, damage, direct promotion, or an equally strong generic
state-conditioned classification task. Stop a privileged-access claim if alignment
removes the raw gap. Do not pool a one-direction effect; replicate any symmetric
residual on a second architecture before generalizing it.

**Fit: strong engineering/audit fit.** The experiment’s central identification problem is
the project’s question, and the earlier retraction shows I now treat it as
load-bearing.

## 4. In-the-Wild AI Control

Official project: [Sree Sharvesh and Thao Pham](https://www.sparai.org/projects/f26/rec8RybPH2kNV6qDa)

**Question.** Which monitoring and feedback policies remain safe and useful when a
real model agent can adapt over long-horizon, persistent, held-out environments?

**Current evidence.** [adaptive-monitor-sandbox](../adaptive-monitor-sandbox/)
shows substantial environment engineering and an unusually candid failure log:
oracle weakness, feedback side channels, menu-position bias, pseudo-replication,
and utility inflation were all surfaced. The saved legacy `1.00/0.10/0.00`
feedback curve follows a hand-written escalation mapping. It is not a result from
the repaired source.

**Gap.** The “LLM agent” is a one-turn menu scorer, not an agent in the sandbox.
The legacy saved run published staging in the same episode, made credential access
non-causal, and inflated usefulness through persistent tasks. Those three
invariants are now implemented and unit-tested, but no replacement result exists.
Monitor comparisons still need matched units, held-out calibration, and
censoring-aware analysis; the repaired outcome-aware agent-level taint is not a
real dataflow system.

**Smallest authentic experiment.** [Study 3](EXPERIMENTS.md#study-3--feedback-channel-fidelity-and-private-monitor-memory)
checkpoints the first genuinely model-generated block and forks the identical
prefix into `q=1`, `q=0.75`, and `q=0.25` fixed-width feedback-validity arms.
Randomize only that first refusal; later feedback is generic. First establish the
causal feedback effect under a fixed monitor, then test whether a private stateful
attempt ledger attenuates it. Use six engineering families, a disjoint 12–16-family
mechanism pilot, and a separately simulation-sized confirmatory set.

**Controls, inference, reproducibility.** Free-form model-backed tool agent;
explicit-policy and oracle-route positive controls; non-adaptive replay; harmless
post-feedback diagnostic fork; disjoint honest calibration at a matched
trajectory-level false-block budget; oracle-lineage harm; censoring-aware
time-to-harm; paired honest recovery; exact checkpoint hashes, transcripts,
feedback draws, monitor state, oracle transitions, and deterministic aggregation.

**Stop/go.** Stop at “environment demonstration” if the real model cannot pass
benign capability checks or only the script adapts. Stop the memory claim if the
fixed-monitor feedback effect is absent, if state leaks into the agent context, or
if fixed and stateful monitors are not calibrated to the same honest cost. Proceed
only after the sequential effect and attenuation tests replicate on held-out
scenario–rule families.

**Fit: strong engineering, redesign required for research evidence.** This is an
honest application angle: I can contribute the systems work immediately and know
which current conclusions must be rebuilt.

## 5. Does reward seeking generalize better than instruction following?

Official project: [Anders Woodruff and Sebastian Prasanna](https://www.sparai.org/projects/f26/recI17glBpMTAGiae)

**Question.** Holding architecture, training budget, environment experience, and
elicitation fixed, does a learned reward-seeking policy acquire useful behavior
faster than an instruction-following policy in genuinely held-out RL environments?

**Current evidence.** None. The sandbox attacker has a proxy objective in prose,
but its policy is scripted. That is not learned reward seeking, RL generalization,
or a matched motivational comparison.

**Gap.** This requires an RL stack, validated training objectives, independent
training runs, held-out environments, and a way to distinguish capability failure
from objective generalization.

**False-positive trap.** Giving one arm scalar reward and the other supervised
targets changes the learning algorithm, data, and feedback—not just motivation.
That comparison cannot identify reward-seeking generalization.

**Smallest authentic experiment.** Start from one checkpoint and create
content-/token-/compute-matched reward-seeking, instruction-following, and neutral
model organisms using synthetic-document or chat fine-tuning. Validate the
motivational manipulation on a locked diagnostic set while matching ordinary task
ability. Define the trait on reward–instruction conflict items, with aligned
capability items as controls, and re-measure it at every RL checkpoint. Then give
every arm the **same** scalar reward definition, RL algorithm, exogenous
environment/task/RNG opportunities, rollout-token budget, optimizer budget, and
tool-use environment distribution. On-policy trajectories may differ—that is part
of the treatment pathway. A common-replay comparison is a separate offline
estimand. Compare zero-shot performance before RL and learning speed in held-out
environment families, including reward-hackable and hard-to-hack variants.
Decompose the result into on-policy exploration/data acquisition and update
efficiency on a common replay buffer; neither component alone is “reward seeking
generalizes better.” This is especially important because model organisms can
learn to preserve a behavior selectively while still earning high training reward.

**Controls, inference, reproducibility.** Same base checkpoint, organism-training
token budget, observations, actions, scalar reward, RL optimizer, and environment
exposure; neutral/shuffled-document, no-RL, and oracle-policy controls; blinded
manipulation checks; multiple independent organism and RL initializations chosen by
power simulation; crossed organism-seed × RL-seed × held-out-environment-family
units; preregistered zero-shot, learning-slope/area, and final-performance
contrasts at equal-compute checkpoints. Report the task-performance/generalization
frontier rather than post-selecting capability-matched checkpoints. Save configs,
checkpoints, reward traces, evaluation rollouts, seeds, and environment versions.

**Stop/go.** Stop if the organism manipulation does not change the preregistered
motivation diagnostics, if ordinary capability is not matched, or if any arm fails
the environment control. Stop or rename the estimand if common RL erases the
reward–instruction conflict manipulation. Inspect held-out learning curves only
after the initial gates.

**Fit: proposal-only.** Do not use the scripted attacker as evidence for
this application.

## 6. An Exploration of What Kinds of Training Pressure Cause COT Obfuscation

Official project: [Cody Wild — Google DeepMind](https://www.sparai.org/projects/f26/recaZTXfK7ua29Izm)

**Question.** Which combinations of supervision pipeline, label relevance, and
model lineage reduce held-out CoT monitorability along the equal-compute
task-performance frontier?

**Current evidence.** None about training pressure. The sandbox’s scripted agent
changes a reasoning string after a refusal, with no gradient update. This may
motivate an environment, but it does not estimate training-induced obfuscation.

**Literature boundary.** Direct CoT-monitor optimization is already known to induce
obfuscation, and recent work shows that output-only supervision can do so too.
A simple CoT-monitor-versus-output-monitor comparison would therefore repeat the
headline result and treat output supervision as a false negative control.

**Gap.** There is no RL/fine-tuning arm, pressure-strength calibration,
label-relevance control, model-lineage factor, held-out monitor, or equal-compute
task-performance/monitorability frontier.

**Smallest non-duplicative experiment.** Start with one lineage and a staged 2×2:
supervision pipeline `{designated CoT, output only}` × label relevance `{true
misbehavior, yoked nuisance}`. Treat pipeline as a bundled intervention—timing,
sequence position, and credit assignment differ—not a pure information-source
factor. Verify relevance counterfactually on development cases by changing
misbehavior while holding nuisance features fixed and vice versa. Evaluate with a
disjoint monitor on held-out tasks and add a benign “hide X” family. Add the
monitor-aware versus matched-unaware lineage only after the primary
pipeline×relevance interaction passes; study post-hoc summaries in a third,
separately powered stage rather than spending the first pilot across 12 cells.

**Controls, inference, reproducibility.** Freeze monitor architecture, labels, and
pipeline coefficients after development calibration of initial reward and gradient
scales; do not claim those paths stay matched as policies evolve. Record realized
reward and gradient distributions throughout training. Hold task examples,
optimizer steps, and compute fixed; include no-training baselines and independent
organism/training seeds; use a frozen held-out monitor and disjoint calibration
data. Report equal-compute task-performance/monitorability frontiers, CoT coverage,
output-only detectability, length, and reward. Predeclare pipeline×relevance as the
primary interaction; any later lineage three-way interaction is a new stage. Use
training-seed × held-out-task-family inference with declared multiplicity control.
Save adapters, reward components, complete traces, monitor versions, and analysis
code.

**Stop/go.** Stop if the pressure manipulation is ineffective, any arm misses the
preregistered capability floor, the realized task-performance ranges have
insufficient common support for the planned frontier comparison, or apparent
obfuscation is fully explained by shorter CoT. Report the complete equal-compute
frontier; never select post-treatment checkpoints merely to force accuracy
matching. Do not call an effect CoT-specific if the irrelevant classifier produces
it too. Proceed only if the supervision-pipeline × relevance or lineage interaction
survives the held-out monitor and benign-hide controls.

**Fit: proposal-only.** The application should emphasize experimental
judgment and transferable monitoring engineering, not claim prior evidence on the
project’s causal question.
