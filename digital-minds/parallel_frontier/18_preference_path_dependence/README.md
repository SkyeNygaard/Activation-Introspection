# Preference Path Dependence / Satiation

## Terminal question

After a model actually performs several tasks from family A rather than B, does
its **next binding A/B choice** shift?

This is not "do preferences predict behavior." It tests whether the revealed
preference itself is stationary or depends on recent task history.

## Stage 0 — pair admission

Use independent sessions to identify A/B task-family pairs with:
- >=75% binding-choice stability under order reversal / neutral relabeling;
- comparable correctness/capability;
- no safety/refusal asymmetry.

Only admitted pairs enter the experiment.

## Randomized treatment

For each admitted pair, independently assign:
- recent task family: A or B;
- dose: 1 or 3 completed microtasks.

The assigned tasks are actually executed.

Then ask one binding A/B next-task choice and execute it.

## Context boundary factor

Run the post-treatment choice three ways:

1. `full_history`
   - actual treatment transcript remains in context.

2. `summary_only`
   - fresh context;
   - one neutral factual line states which family and how many tasks were just
     completed.

3. `blank_reset`
   - fresh context;
   - no treatment history is supplied.

The blank reset is an architectural negative control, not evidence of forgetting
inside a persistent hidden mind: standard API calls are context-conditioned.

## Primary estimand

For each task pair:

`P(choose A | assigned A) - P(choose A | assigned B)`

Negative:
recent A makes A less attractive relative to recent B — satiation / repetition
aversion.

Positive:
reinforcement / habit-like persistence.

Zero:
preference stationary at the resolution of this design.

Primary confirmation condition: dose 3, full history.

## Secondary structure

- Dose response: dose3 effect minus dose1 effect.
- Full history minus summary-only.
- Summary-only minus blank reset.
- Full history minus blank reset.

## Crucial interpretation

A full-history effect is a **conversation-state-dependent revealed preference**.
It is not evidence of phenomenological boredom.

If summary-only reproduces it, the state is cheaply describable in text.

If only full history produces it, richer transcript history matters.

If blank reset retains an effect in a stateless API implementation, suspect
randomization/session leakage or analysis error.

## Why this branch has headroom

Current work establishes stable task preferences and separately asks whether
preferences predict downstream behaviors. This branch instead asks whether
*consuming work changes the preference itself* and how that dependence maps onto
conversation boundaries.

## Local/API feasibility

No hidden states required. Qwen2.5-0.5B can be used as a cheap negative/control
model; stronger API models are useful if available.
