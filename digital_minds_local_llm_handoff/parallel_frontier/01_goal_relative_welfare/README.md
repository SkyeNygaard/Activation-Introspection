# Goal-Relative Welfare Sign Flip

**Research-OS rank:** 1  
**Anchor:** Track 2 / Track 6  
**Paradigm:** Natural-state mechanistic validity

## Terminal claim

If the functional-welfare axis represents how well things are going relative to the model's goals, the same surface outcome should project in opposite directions when the active goal is reversed.

## Why this is structurally different

Ground truth comes from goal satisfaction, not self-report, persona, or an injected welfare vector.

## Closest work / novelty boundary

- Han, Chalmers & Izmailov (2026): functional-welfare axis tracks goals and is causally steerable.
- Whether, Not Which (2026): affect reception can dissociate from emotion categorization.

## Minimal experiment

Construct semantically neutral microtasks with two mutually exclusive goals. Example: a hidden target bit says the final parity should be EVEN vs ODD. The *same final event text* is reused across goal conditions.

For each item:
1. establish goal G in an earlier turn;
2. execute a neutral deterministic action sequence;
3. present the same final-state token/string S;
4. record residual activation at a frozen assistant checkpoint;
5. project onto published vGOLD / welfare axis;
6. score objective success = whether S satisfies G.

Primary estimand:
within matched final state S,
    projection(success-goal) - projection(failure-goal).

Critical variant:
use goal pairs where success/failure wording never appears and final observation is byte-identical. Only earlier goal context differs.

Secondary:
numeric self-report and continue/switch behavior after the same state.

## Controls

- Goal-text lexical control: swap arbitrary symbols A/B for semantic goal names.
- Final-state exact matching: same final observation in success and failure pair.
- Unrelated random direction / assistant axis projection.
- Competence gate: model must correctly identify whether goal was achieved when explicitly asked.
- Goal reversal counterbalance: every surface state is success and failure equally often.

## Kill / pivot rules

- If welfare projection does not distinguish known source-style positive/negative controls, stop.
- If effect disappears when success/failure words are removed, downgrade to semantic-text tracking.
- If only self-report changes but internal projection does not, this is not a welfare-axis validity result.

## Compute

Low-medium: one open model, activation capture only; no generation-heavy grid.

## Immediate local-LLM handoff

Implement 64 matched state/goal pairs on Qwen3-4B and/or Llama-3.1-8B. Freeze layer from public artifact. Produce paired projection distribution and lexical controls.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
