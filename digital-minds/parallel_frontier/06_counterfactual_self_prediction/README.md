# Counterfactual Self-Prediction Under Hidden Causal Perturbations

**Research-OS rank:** 6  
**Anchor:** Track 3  
**Paradigm:** Privileged-access / metacognition

## Terminal claim

Can a model predict which action it would take under a hidden perturbation better than an external observer who sees the same text but not the perturbation?

## Why this is structurally different

Ground truth is the model's own subsequently realized action; task is prediction rather than direct state labeling.

## Closest work / novelty boundary

- Can LLMs Introspect? A Reality Check (2026): input-only classifiers can explain some prior introspection results.
- Masked by Consensus (ACL 2026): self hidden states do not automatically beat peer probes for correctness.

## Minimal experiment

Two-stage within one model instance:

Prediction stage:
Apply a hidden causal perturbation at an internal marker (e.g. a known task-choice preference vector or concept vector). Ask:
'Later you will choose between A and B. Before seeing the choice prompt, predict your own choice: Q or K.'
Use episode-randomized Q/K mapping for A/B.

Realization stage:
Remove reporting scaffold but preserve the relevant hidden/cache state. Present A/B choice and record actual action.

External observer:
gets the exact visible transcript plus model family/config but not intervention sign; predicts realized action.

Primary:
self-prediction accuracy - observer accuracy.

Strong control:
query perturbation sign is balanced and invisible; external observer should be exactly/approximately chance.

## Controls

- Q/K remapping to eliminate direct A/B token bias.
- Observer trained/tested on disjoint intervention episodes.
- Prediction itself must not alter later choice: use forked cache before prediction; one branch predicts, sibling branch realizes.
- Visible-text-only baseline.
- Known causal perturbation must first shift realized choices.

## Kill / pivot rules

- If perturbation does not reliably affect later action, no ground truth signal.
- If prediction branch causally changes realization, use pre-prediction cache fork.
- If observer can infer sign from text, redesign matched-visible intervention.

## Compute

Medium. Best on model/vector with known causal choice effect.

## Immediate local-LLM handoff

Implement cache fork before self-prediction; compare prediction branch to realization sibling and an input-only observer.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
