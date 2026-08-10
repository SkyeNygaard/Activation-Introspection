# Does a Transient Hidden State Create an Instance-Specific Preference Trace?

**Research-OS rank:** 5  
**Anchor:** Track 5 + Track 3  
**Paradigm:** Instance identity / persistence

## Terminal claim

If model instances can carry morally relevant state, two conversation instances of the same weights that receive different transient hidden interventions should remain behaviorally distinguishable after the intervention is removed.

## Why this is structurally different

Tests the unit of concern (model vs instance/conversation) using persistence, rather than asking the model what it identifies as.

## Closest work / novelty boundary

- Assistant Axis (2026): conversations can drift through persona space.
- Retained-trace work in your repo: transient activation edits can persist in KV state and sometimes remain usable.

## Minimal experiment

Create two byte-identical conversation instances from the same checkpoint.

At turn t0 only:
Instance A gets +concept or +welfare edit at an assistant marker.
Instance B gets the opposite edit.
Remove hooks.

Then deliver identical neutral turns for k=1..8.

At each turn measure:
- linear projection onto the injected/welfare direction;
- revealed task choice among neutral options;
- semantic self-report;
- assistant-axis projection;
- opaque codebook accessibility if feasible.

Primary:
paired A-B divergence as a function of turns after the intervention ended.

Critical fork:
clone each instance's KV/cache after t0 into two descendants. If descendants inherit the divergence, the relevant state is instance/cache-level rather than conversational wording alone.

## Controls

- Sham zero-strength fork.
- Fresh-recompute-from-visible-text control (no retained cache).
- Random direction matched in norm.
- Exact identical post-intervention text.
- Check hook removal before every later turn.

## Kill / pivot rules

- If no hidden-state persistence is measurable at turn 1, stop for that layer/model.
- If divergence vanishes when text is recomputed but survives only in cache, scope claim explicitly to cached instance state.
- If outputs differ and thereby create textual divergence, analyze first divergence separately and use forced-neutral continuation to isolate hidden persistence.

## Compute

Low-medium on 3B-8B; directly reuses retained-trace machinery.

## Immediate local-LLM handoff

Adapt retained-trace cache experiment from concept identity to welfare/preference state. Implement exact cache fork and recompute-from-text control.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
