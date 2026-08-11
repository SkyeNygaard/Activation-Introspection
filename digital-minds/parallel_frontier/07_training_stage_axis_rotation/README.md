# What Post-Training Changes: Axis Rotation vs Causal Coupling

**Research-OS rank:** 7  
**Anchor:** Track 2 + Track 5  
**Paradigm:** Training-stage comparative representation

## Terminal claim

Pretrained and post-trained models may share welfare/persona directions geometrically while differing in how strongly downstream behavior is causally coupled to them.

## Why this is structurally different

Tests training-induced causal coupling across matched checkpoints, not persona prompting or report reliability.

## Closest work / novelty boundary

- Functional Welfare Axis (2026): RL recruits rather than creates a pre-existing axis.
- Assistant Axis (2026): assistant-like axis exists in pretrained counterparts too.

## Minimal experiment

Choose a model family with matched base and instruct/post-trained checkpoints and compatible public axes.

For welfare and Assistant axes separately:
1. estimate/extract direction in base and post-trained model;
2. align layers by relative depth;
3. measure cross-checkpoint cosine/CCA geometry;
4. apply each direction to its own and counterpart checkpoint;
5. measure causal effects on task choice, refusal, self-description, and neutral goal-success probes.

Key decomposition:
Geometry persistence = direction remains similar.
Causal recruitment = same-sized projection/intervention has stronger behavioral effect post-training.

Primary:
difference in causal slope d(behavior)/d(axis projection) base vs post-trained, with direction geometry reported separately.

## Controls

- Same tokenizer/model family if possible.
- Layer-relative alignment and norm/damage calibration.
- Random directions.
- Base-model completion formatting rather than chat-only prompts.
- Separate geometry from behavior; never infer recruitment from cosine alone.

## Kill / pivot rules

- If checkpoints are architecture/tokenizer incompatible enough to prevent aligned interventions, choose another family.
- If base model cannot perform downstream tasks, use logit-level markers rather than chat behavior.
- Do not duplicate source paper's exact maze experiment.

## Compute

Medium-high but parallelizable; 2 checkpoints × small layer set.

## Immediate local-LLM handoff

Find matched base/instruct family with public welfare or assistant vectors. Build geometry + causal-slope table, not a broad behavioral benchmark.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
