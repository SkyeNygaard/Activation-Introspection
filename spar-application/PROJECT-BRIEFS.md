# Six project briefs

Each brief answers the same six questions: what I would study, what evidence I
currently have, what is missing, the smallest authentic portfolio experiment,
the controls/inference/reproducibility package, and the condition under which I
would stop or proceed.

**Original research is being done for projects 1 and 3 only.** The other four
briefs are written from work already finished and are not receiving new runs. The
reasoning, the shared problem statement behind 1 and 3, and what is being tried
next are in [RESEARCH-DIRECTION.md](RESEARCH-DIRECTION.md).

## 1. Introspection Training for Verbalization Activations

Official project: [Belinda Li, Anthropic](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz)

**Question.** Can training make a model’s verbalizations track causally introduced
activation content on held-out concepts, rather than visible prompt cues or generic
perturbation artifacts? The safety target is reliable monitoring under shortcut
pressure, not self-description for its own sake.

**Current evidence.** [activation-introspection](../activation-introspection/)
now contains an executed controlled instance of the project's zero-training
question. I selected Qwen2.5-3B layer 9 and strength 1.0 on DEV. After auditing an
initial artifact, I froze a repaired confirmation without changing those settings
or the gates, then ran eight fresh concept directions crossed with three fixed
carrier strings. Four causally edited hidden-state demonstrations teach an
arbitrary `Q/K` mapping while visible observation content is held fixed: target
accuracy is 0.891 [0.816, 0.995], against 0.500 for the exactly query-matched arm.
The target beats the strongest per-position-magnitude-matched random/shuffled
direction by +0.231 [0.137, 0.286], with 1.000 next-token label-format integrity.

The strongest property of this design is structural rather than statistical.
Because query twins are byte-identical in visible text and carry opposite correct
labels, an input-only learner is pinned at exactly 0.500 **by construction**. The
"reality check" objection — that input-only classifiers can match hidden-state
prediction — cannot apply here, because there is no input to classify on. That is
the confound prior activation-label ICL work leaves open.

The effect then replicated out of bank. Three concepts and two carriers used in no
tuning or selection step give 0.958 target accuracy against 0.500 for the exactly
query-matched arm, correct sign in 6/6 strata. This came from the control arm of a
later experiment run for a different purpose, so it is an unplanned generalization
check rather than a confirmatory rerun.

The earlier retained-trace replication supplies the complementary failure case.
A naturally trained probe can recover a trace at readout depth after the model's
own behavioral access has collapsed. Together, the results identify a usable
in-context interface and a later-depth regime that training would have to repair.

**Trained evidence.** A LoRA fitted on eight concept directions reports the sign
of a causally injected hidden state at **0.927 mean twin-pair accuracy, range
[0.833, 1.000] across four training seeds**, on eight directions and three
carriers withheld from training. The untrained base model is at 0.000 on every
seed and magnitude-matched controls at 0.260/0.208, with format and label mass
1.000 throughout. The pair-wise null is 0.000 for a prompt-only strategy, so the
frozen 0.500 threshold is conservative.

Two failed versions are the more useful half. V1 scored 0.917 and passed every
gate while holding 5e-9 probability on the answer tokens: the loss was a two-way
softmax over the label logits, which fixes their ordering and leaves the rest of
the vocabulary free, so the adapter suppressed both labels and kept the right one
on top. **Restricting an introspection-training loss to the answer options
produces a probe wearing the model's output head, and no forced-choice metric can
see it.** V2 repaired the loss, scored 0.583, and I reported that as the cost of
the repair — also wrong, because neither run seeded the adapter's initialization.
Both traps sit directly in this project's path.

**The trade-off that training creates.** A further study trains two adapters that
differ only in whether the label convention is fixed or re-randomised, then scores
both against the untrained model at and below the training strength. Two findings,
three seeds, all gates passing.

Training extends the detection floor: at injection strength 0.15 the untrained
model is at exactly 0.500 with 0.010 twin-pair accuracy — blind — while adapters
trained only at 0.5 read it at 0.790–0.863. And training destroys selectivity:
untrained, magnitude-matched random directions sit at chance (0.513) while concept
directions reach 0.745, but after training random directions reach 0.913–0.955.

> Introspection training buys sensitivity and pays for it in specificity.

A monitor trained this way answers "did something move at this site" rather than
"is concept X active", so any displacement — including an adversarial one with no
semantic content — yields a positive report. That is a concrete failure mode for
activation monitoring, and it is the most directly project-relevant result here.
It also settles the probe objection: both adapters score 1.000 on mapping-flip
pairs, where a fixed sign-to-token readout scores 0.000 by construction.

**Gap.** Every edit in every study here is an injected direction. Nothing yet
shows the model reporting a state it computed on its own, which is the largest
distance between this work and the project's eventual target. Everything is one
model, one layer, binary answers; three or four seeds give a mean and a range,
not an interval. The causal ICL result is binary and
explicitly elicited. Neither isolates privileged access, uses J-space variables,
or shows free-form verbalization. The retained-trace schedule remains a
replication. Nothing here yet answers whether training generalizes across
internal variables, layers, or naturally occurring computations.

**Smallest authentic experiment.** Seeds and episode-remapped training are done.
The next step follows from the trade-off above: test whether a trained reporter
transfers to a **naturally computed** internal state rather than an injected one.
Locate an intermediate the model computes on its own — a two-hop bridge entity is
convenient — verify by patching that it is causally load-bearing downstream, then
swap it between matched prompts with the visible text held identical and elicit
the report before the answer appears. Compare base, fixed and remap adapters.
The design is only worth running if it can separate "the capability does not
transfer" from "the instrument did not reach", so the patch needs a positive
control on downstream behaviour and the reporter needs a within-regime anchor.
Then extend to unused layers and to richer multiway/continuous variables. Keep
the unrestricted full-vocabulary format and label-mass gates on every arm; they
are what caught V1. A
later J-space phase should replace lexical contrast vectors with causally patched
workspace variables and ask whether the learned verbalizer transfers. Keep the
symmetric aligned-source experiment as the privileged-access test; do not make it
the first expensive run.

Two inference-only feasibility attempts now close one narrow instrument. An exact
transplant at the route marker after layer 9 did not make the ordinary two-hop
answer follow the donor bidirectionally in any of five worlds
([`notes/09`](../activation-introspection/notes/09-natural-state-pilot.md)). Its
successor removed the two obvious explanations — it used a task the model solves
10/10 at probability 1.000, transplanted the output-ready state at the position
that produces the answer, and screened three prospectively named layers — and
recovery was +0.001, −0.003 and +0.100 against a frozen 0.5 threshold, with 0/5
tasks controlled at every layer
([`notes/10`](../activation-introspection/notes/10-output-ready-arithmetic.md)).

Neither is a natural-state reporting null: the frozen stop rule prevented the
reporter from running both times. A post-hoc development-bank localization over
all 36 blocks then found why. Through block 26 the pre-answer state does not
favour its own answer over its twin's better than chance under a logit lens, and
the twin states differ by under a quarter of the residual norm — there is nothing
to transplant. From **block 27** the identical intervention makes the donor's
digit the full-vocabulary argmax in 10/10 transplants at recovery 0.787, which
clears the frozen gate outright. The anchors, drawn from the layers where
*planted* directions are readable, sat one block below where the *computed*
answer appears.

A third run froze block 27 as a disclosed post-hoc site, changed nothing else, and
required it to pass the same gate on the never-scored held-out bank. Development
reproduced at 5/5 and recovery 0.787. The held-out bank returned **3/5** against
a frozen 4/5, so the reporter did not run and the protocol forbids reselecting a
layer. Eight of ten held-out transplants worked; pooled with development that is
a per-transplant rate of 0.90, at which a five-task "4/5 in both directions"
criterion fails about a quarter of the time by itself. The binding defect is
therefore bank size, not the site.

Gate 3 — a causally reachable natural state — is not cleared, and the path to it
is now specific rather than open-ended: ten or twenty twin pairs instead of five
so the gate measures a rate instead of quantising it, and a donor column across
blocks 27–33 at the one token instead of a single block, since recovery plateaus
at 0.78–0.80 there in a way that a partially re-asserted computation would
produce. Both need a fresh protocol and a third bank; the two existing banks are
spent.

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
self-specific compatibility under the tested transform, not metacognition.

**Fit: an executed in-context benchmark and an executed training result, plus a
methodological trap found the hard way.** The ICL benchmark exists and passed its
stated gates; the trained reporter generalizes to unseen directions across four
seeds; and the sensitivity/specificity trade-off is a concrete failure mode for
activation monitoring. The three failures on the way — a loss restricted to the
answer options, a number quoted from an unseeded run, and a hypothesis falsified
by its own gates — are contributions to how this project should be evaluated.
Generalization to naturally occurring internal variables remains untested and is
the direction I would propose.

## 2. Deploying Programmatic Attention in Real Transformers

Official project: [Belinda Li, Anthropic](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L)

**Question.** The project asks whether programmatic QK circuits can run inside
real, usable transformers at minimal cost to both task performance and
efficiency. I read that as two coupled questions and have partial evidence on
each: can a human-readable QK program constrain the causal source of an
activation report, and does an exact programmatic lowering actually pay off once
it is inside a real attention module?

**Current evidence, routing side — a pre-registered negative.** A frozen
one-concept DEV screen (Stage 1a) selected query-marker L21/L23 and final-answer
L26/L31. A second frozen screen (Stage 1b) tested all 64 layer-role-head
components on three disjoint concepts and two carriers, 5,112 scored forwards,
with the analyzer hash-locked before any output was inspected. It **failed its
own gate**: `query_marker@23` removed 16.2% against a 20% parent threshold, and
six components qualified against a frozen 2–4 window, with the three
`final_answer@26` heads jointly exceeding 110% of their parent's effect. Influence
is redundant and broader than a compact route, so the fixed-route program study
does not proceed. The screen also lacked a zero/random-donor damage control, which
is a gate-ordering error I would fix by moving that control inside the screen.

**Current evidence, efficiency side — an informative collapse.** I lowered one
released GPT-2 positional program to an exact `O(TD)` form, eliminating the `T×T`
matrix, Q/K, and softmax for that head. Equivalence holds across 216 CPU fp32
cells at max abs error 4.8e-7. As an isolated operator at `B=1, T=1024` it is
18.63× [18.45, 18.83] faster; integrated into the real GPT-2 attention module via
native head pruning it is 1.089× [1.088, 1.091], missing a 1.25× threshold frozen
before the grid ran. The algebra was never the bottleneck — partial-head
projection and dispatch are. That is a concrete answer to the project's cost
question and a specific target for the Triton/CUDA workstream.

**Gap.** No result shows that readable QK routing improves activation-monitor
provenance, excludes alternative information paths, or remains faithful when a
visible cue conflicts with the hidden state — and the localization stage that
would have made that testable stopped. On the systems side, everything is CPU,
forward-only, uncached, one program, one model, with no allocator peak-memory
measurement and no behavioral-fidelity evaluation of the pruned model.

**Literature boundary.** *Explaining Attention with Program Synthesis* already
generates executable attention programs, replaces heads in GPT-2/TinyLlama/Llama,
and measures perplexity and downstream behavior. Replacing one head and showing
that the model still works would be a replication, not a new deployment result.

**Smallest non-duplicative experiment.** Stages 1a and 1b are executed; 1b
stopped. The next honest step is not to re-run the same screen hoping for a
different verdict. It is to fix the two things that made the stop uninformative
about mechanism: add zero-donor and random-donor patches at each component so
removal fraction can be separated from disruption, and widen the component
universe past single query heads at preselected sites to include head pairs and
the KV side. If influence is still redundant under those conditions, the
right conclusion is that this behavior is not a fixed-route target, and the
program should be imposed during training rather than discovered after it.

On the systems side the smallest next experiment is sharply defined by the 1.089×
result: measure where the integrated overhead goes between projection, dispatch,
and the residual dense path, then test whether a fused partial-head projection
recovers any of the isolated 18.63×. That is a small, falsifiable inference-
workstream contribution with a preregistered threshold.

Stage 2, if a compact route ever replicates, adds a decoy after the edited marker. Make it predictive during reporter
training, then independent or reversed at test. Compare matched reporters that
differ only in QK routing: learned attention, a fixed `query → marker` program,
wrong-marker/random gather, and programmed-edge ablation. Keep V/O and the output
readout matched, and make the declared head the reporter's only ingress from the
subject trace. Test unused concepts, prompt families, mappings, and lengths.

**Controls, inference, reproducibility.** Exact hidden-state × cue factorial;
input-only, query-only, wrong-marker, random-gather, and edge-ablation controls;
multiple independent reporter-training seeds; crossed concept/prompt inference;
saved logits, routes, checkpoints, protocol, and raw rows. Primary outcomes are
the hidden-intervention effect, cue-swap effect at fixed hidden state, reversed-cue
worst-group accuracy, and the fraction of the hidden-state effect removed by
ablating the declared edge. Latency and peak memory are reported as a separate
outcome family with their own preregistered thresholds, not folded into the
safety verdict: a runtime gain cannot rescue a failed faithfulness gate, and a
faithfulness gain does not excuse an unusable module.

**Stop/go.** The Stage 1 stop rule was "stop if influence is diffuse," and on
2026-08-10 it fired. That decision is recorded rather than revised. Proceeding
with a safety claim would still require a program-routed reporter that retains a
preregistered hidden-state effect, has cue sensitivity inside an equivalence
bound, and loses the effect when its declared route is ablated. Runtime gains
cannot rescue a failed safety gate, and the 1.089× integration result cannot be
promoted into one either.

**Fit: two executed results, both negative against their own thresholds, plus a
protocol the negatives constrain.** I can supply hook-level transformer
modification, an exact program lowering with verified equivalence, and a
measured localization of where integration cost actually sits. What I do not have
is a compact route to program or a shortcut-robustness result, and the reason is
recorded with its evidence rather than left as a gap.

## 3. Faithfulness, Self-Knowledge, and Introspection

Official project: [Noah Siegel, Google DeepMind](https://www.sparai.org/projects/f26/rec3KQAI0JcxJJAce)

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
intervention cannot have promoted the correct answer token, which is exactly the
artifact that produced a fake 100% result earlier in this repo. And the two
do-nothing arms land on exactly chance at every site, so the visible prompt
provably carries nothing.

The pilot also exposed why this distinction is hard in the first place: a probe
measured at one causal site was compared against behavior at another, and the
resulting headline had to be retracted.

**Gap, now closed — and the answer is negative.** That control has been run. On
the identical 576 episodes behind the 0.891, a four-shot nearest-centroid reader
given the same five post-injection residual states scores **1.000** while the
model scores 0.892, with **62 reader-only episodes and zero model-only episodes**.
Under the field's operative criterion ([arXiv 2508.14802](https://arxiv.org/abs/2508.14802))
there is no privileged access at this site: the model reads its own state worse
than an outsider does. The structural contribution of the design survives — an
input-only learner is still pinned at exactly 0.500 by construction — but the
introspection reading of the number does not. See
[notes/11](../activation-introspection/notes/11-matched-cost-reader.md).

This is the result that most directly serves this project, because it is the
project's own question answered with a measurement rather than an argument, and
because the 62 failures localize the interesting phenomenon: information linearly
present in the residual stream that the model itself does not use. The study is also a
replication of a published effect, and covers one model family.

**Smallest authentic experiment.** [Study 1](EXPERIMENTS.md#study-1-causal-use-of-a-retained-activation-trace-executed-2026-08-01)
is done and serves as the reporting-interface gate. [Study 2](EXPERIMENTS.md#study-2-does-an-own-model-advantage-survive-representational-alignment)
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

**Smallest authentic experiment.** [Study 3](EXPERIMENTS.md#study-3-feedback-channel-fidelity-and-private-monitor-memory)
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
targets changes the learning algorithm, data, and feedback, not just motivation.
That comparison cannot identify reward-seeking generalization.

**Smallest authentic experiment.** Start from one checkpoint and create
content-/token-/compute-matched reward-seeking, instruction-following, and neutral
model organisms using synthetic-document or chat fine-tuning. Validate the
motivational manipulation on a locked diagnostic set while matching ordinary task
ability. Define the trait on reward–instruction conflict items, with aligned
capability items as controls, and re-measure it at every RL checkpoint. Then give
every arm the **same** scalar reward definition, RL algorithm, exogenous
environment/task/RNG opportunities, rollout-token budget, optimizer budget, and
tool-use environment distribution. On-policy trajectories may differ, and that is part
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

Official project: [Cody Wild, Google DeepMind](https://www.sparai.org/projects/f26/recaZTXfK7ua29Izm)

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
misbehavior, yoked nuisance}`. Treat pipeline as a bundled intervention, because timing,
sequence position, and credit assignment all differ. It is not a pure
information-source factor. Verify relevance counterfactually on development
cases by changing
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
