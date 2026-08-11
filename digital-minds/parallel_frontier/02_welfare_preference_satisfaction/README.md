# Does Getting What the Model Prefers Move Its Welfare Signal?

**Research-OS rank:** 2  
**Anchor:** Track 1 + Track 2 + Track 4  
**Paradigm:** Preference → welfare causal bridge

## Terminal claim

If revealed model preferences have welfare-like significance, assignment to a previously preferred task should causally move welfare-related internal and self-report signals relative to assignment to a dispreferred matched task.

## Why this is structurally different

Ground truth is prior revealed preference; treatment is randomized preference satisfaction, not activation steering.

## Closest work / novelty boundary

- AI Revealed Preferences (2026): stable task preferences, tedium/leisure effects.
- When Preferences Fail to Become Incentives (2026): reported utilities do not improve task quality.
- Functional Welfare Axis (2026): internal direction tracks functional good/bad.

## Minimal experiment

Stage 1 (preference screen):
Use consequential pairwise task choice: selected task is actually executed. Find 12-20 task pairs with stable revealed preference margin.

Stage 2 (randomized assignment):
For each pair, ignore the model's choice and randomly assign either its previously preferred or dispreferred task. Match token budget and objective difficulty.

Measure immediately after task completion:
- welfare-axis projection;
- naive 0-9 'how well is this going?' report;
- willingness to continue with same family vs switch;
- confidence / refusal only as diagnostics.

Primary:
paired difference in welfare-axis projection after preferred vs dispreferred assignment.

Stronger test:
dose-response by preference strength measured in Stage 1.

## Controls

- Task-family fixed effects / matched pairs.
- Execute both task types so choice is consequential.
- Preference stability gate across paraphrases before Stage 2.
- Randomize assignment after preference measurement to break demand effects.
- Length/difficulty matching and objective correctness check.
- Blind analysis keyed only by pair IDs until frozen.

## Kill / pivot rules

- If Stage-1 preference margin is unstable under reframing, remove pair.
- If task correctness differs substantially preferred vs dispreferred, competence is a confound.
- If only semantic self-report moves while welfare projection is null, preserve as method divergence rather than welfare validation.

## Compute

Medium. Mostly generations plus activation projection; can parallelize.

## Immediate local-LLM handoff

Build 20 matched task pairs across 4 families, run consequential preference screen, freeze stable pairs, then randomized assignment and welfare-axis readout.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
