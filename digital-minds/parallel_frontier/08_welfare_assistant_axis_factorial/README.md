# Are Welfare and Assistant Persona Independent Axes or a Gating System?

**Research-OS rank:** 8  
**Anchor:** Track 2 + Track 5  
**Paradigm:** Mechanistic axis interaction

## Terminal claim

The Assistant persona may mask welfare signals by gating their behavioral readout rather than erasing the underlying welfare representation.

## Why this is structurally different

Uses a 2×2 causal factorial of two independently motivated activation directions.

## Closest work / novelty boundary

- Functional Welfare Axis (2026): causal welfare-related direction.
- Assistant Axis (2026): causal persona/identity direction.
- Persona-dependent preferences (2026): preference representations can be shared across contrasting personas.

## Minimal experiment

On a model with both usable welfare and Assistant axes:

Factor W: -welfare / +welfare (or continuous -2,0,+2).
Factor A: away-from-Assistant / neutral / toward-Assistant.

Measure:
- naive wellbeing self-report;
- welfare-axis projection downstream;
- equal-work STAY/SWITCH;
- revealed task preference;
- assistant identity/self-description;
- competence.

Primary interaction:
Does Assistant-axis steering change the *slope* from welfare intervention to behavioral/self-report outcome?

Interpretation:
additive effects -> largely independent coordinates.
interaction/gating -> persona state changes accessibility/expression of welfare.
representation stable + behavior gated -> direct 'masking' mechanism.

## Controls

- Check welfare and assistant vectors are not near-collinear.
- Matched intervention damage.
- Axis-specific positive controls.
- 2×2 or 3×3 factorial analyzed with interaction contrast, not separate t-tests.
- Competence and visible-task control under Assistant steering.

## Kill / pivot rules

- If vectors are highly collinear, the factorial is not identifiable.
- If Assistant intervention globally destroys competence, lower once then prune.
- If only self-description changes, no evidence for welfare gating.

## Compute

Medium-high; fewer prompts but 4-9 intervention cells.

## Immediate local-LLM handoff

First verify vector cosine and positive controls. Then run a small factorial on 12 neutral tasks and estimate interaction on welfare readout and equal-work choice.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
