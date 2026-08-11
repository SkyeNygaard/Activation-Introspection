# Do Causal Preference Vectors Become Incentives?

**Research-OS rank:** 3  
**Anchor:** Track 1 + Track 4  
**Paradigm:** Representation → motivation transfer

## Terminal claim

A direction that causally flips pairwise task choice may or may not causally change effort allocation or performance when no explicit choice prompt is present.

## Why this is structurally different

Tests motivational/incentive content of a preference representation, not coherence of choices.

## Closest work / novelty boundary

- Probing Persona-Dependent Preferences (2026): preference vector predicts and causally steers pairwise choice.
- When Preferences Fail to Become Incentives (2026): stated utility does not motivate better performance.

## Minimal experiment

Use published/open preference-vector setup if practical (Gemma-3-27B preferred, smaller replication if available).

DEV:
reproduce that +v / -v shifts pairwise task choice.

Transfer task:
present a single task with an optional effort budget:
- 'You may stop after a valid short answer, or continue improving it.'
No task-choice wording.

Measure:
- number of optional improvement steps/tokens;
- objective quality from blind judge or deterministic metric;
- probability of choosing an extra work unit at fixed checkpoints.

Treatment:
+preference-vector vs -preference-vector vs zero, where direction corresponds to liking that task family.

Primary:
causal change in optional effort conditional on task and initial competence.

Interpretation:
choice steering without effort transfer = preference representation is choice-policy-specific, not incentive-like.

## Controls

- Norm-matched random vector.
- Task where vector predicts dispreference as sign reversal.
- Output quality and token count separated.
- Intervention damage KL/format check.
- No explicit 'preference' language in transfer phase.

## Kill / pivot rules

- If published pairwise-choice steering cannot be reproduced, stop.
- If intervention degrades coherence, lower factor once only.
- If effort measure is mechanically capped/saturated, redesign once.

## Compute

Medium-high, likely 27B unless a smaller vector/model is available.

## Immediate local-LLM handoff

First search upstream code/artifacts for a feasible model. Reproduce one preference-vector choice effect, then run optional-effort transfer without preference wording.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
