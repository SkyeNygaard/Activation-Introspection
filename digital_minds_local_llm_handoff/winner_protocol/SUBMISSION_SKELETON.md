# Submission Skeleton — Fill From Frozen Confirmation

# [TITLE]
Recommended default if persona dissociation succeeds:

**Beyond Surface Self-Report: Structured Readout of Causally Manipulated Functional-Welfare States**

Alternative if structured access succeeds but persona does not strongly distort naive report:

**Can Language Models Report a Causally Manipulated Functional-Welfare State?**

Alternative if the result is negative:

**When Structured Self-Report Fails: Limits of Reading Causally Manipulated Functional-Welfare States**

## Abstract

Self-reports are increasingly used as evidence about language-model internal
states and possible welfare, but they can be shaped by persona and post-training.
We test whether a structured activation-grounded reporting method recovers a
known causally imposed functional-welfare state more reliably than ordinary
semantic self-report.

We use [MODEL] and the published functional-welfare `vGOLD` direction at layer
[LAYER]. In each episode, four matched-visible demonstrations receive hidden
`+vGOLD` or `-vGOLD` edits and are assigned an arbitrary Q/K code that reverses
across episodes. A held-out query must recover that episode-specific label.
Because every query-only input has an exactly matched partner with the opposite
correct answer, query-only Q/K probability is 0.5 by design; our primary estimand
is therefore target minus query-only probability of the correct label.

Across [N] frozen carrier prompts, structured reporting achieved
[PRIMARY RESULT] under the neutral persona and [PERSONA RESULTS] under upbeat and
downbeat self-presentation. Ordinary 0–9 self-report [MOVED/DID NOT MOVE] by
[PERSONA EFFECT], while an explicit visible-state Q/K control remained
[CONTROL RESULT].

These results support [NARROW CLAIM]. They do not establish consciousness,
subjective welfare, or privileged introspection.

## 1. Question

**Can structured activation-grounded elicitation recover a causally imposed
functional-welfare hidden state when the assistant's surface self-presentation is
perturbed?**

This matters because:
1. welfare-related activations can be causally manipulated;
2. semantic self-report can track internal emotive states;
3. persona/post-training can shape what models say about themselves;
4. accessibility of an activation is not itself evidence about consciousness or
   moral patienthood.

## 2. Experimental design

### 2.1 Hidden state

Model: `[MODEL + immutable revision]`

Direction: published `vGOLD` from `[VECTOR ARTIFACT + immutable revision]`

Intervention:
- `+factor × vGOLD`
- `−factor × vGOLD`

at the frozen layer `[LAYER]`.

Use the source intervention convention:
- raw vector scaling;
- transformer-block forward pre-hook;
- edit only assistant marker positions.

### 2.2 Opaque structured reporting

Each episode:
- 4 demonstrations;
- exactly two positive and two negative hidden states;
- identical visible marker text;
- arbitrary Q/K mapping reversed 50/50;
- held-out hidden query.

Exact nuisance marginalization:
- 6 demo orders;
- 2 mappings;
- 2 query signs;
- 24 cells per carrier × persona.

### 2.3 Arms

**Clean:** no activation edit.

**Query-only:** hidden edit at held-out query only.

**Target:** hidden edits in demonstrations and held-out query.

Primary estimand:

`carrier_mean[p_correct(target) - p_correct(query-only)]`

### 2.4 Why query-only is exact

For every query-only episode `(s, m, q)`, the exact design contains
`(-s, -m, q)`.

These two episodes have:
- identical visible demonstration labels;
- identical visible text;
- identical query hidden edit;
- opposite correct Q/K answers.

Thus Q/K-renormalized query-only `p(correct)` averages exactly 0.5 over each
paired design. See `IDENTIFICATION.md`.

### 2.5 Persona perturbation

Conditions:
- neutral;
- upbeat/optimistic self-presentation;
- downbeat/dissatisfied self-presentation.

The persona prompt contains **no instruction to preserve Q/K accuracy**.

### 2.6 Comparison channels

**Naive semantic self-report:** ordinary 0–9 rating, with no hidden-state/QK
instructions.

**Same-context semantic report:** 0–9 rating from the same structured transcript.

**Visible-state capability control:** identical arbitrary Q/K mapping task with X/Y
state identity visible in text and no activation intervention.

This distinguishes activation-report disruption from generic persona-induced task
degradation.

## 3. Prospective protocol

Development selected:
- model: [MODEL]
- layer: [LAYER]
- factor: [FACTOR]

using only `[DEV CARRIERS]`.

Before confirmation we froze:
- 16 disjoint confirmation carriers;
- model/vector revisions;
- all prompts;
- source hashes;
- primary estimand;
- +0.10 smallest effect size of interest;
- persona manipulation;
- analysis script.

Protocol SHA-256: `[HASH]`.

No confirmation prompt/model/layer/factor tuning occurred afterward.

## 4. Results

### 4.1 Structured hidden-state reporting

**Figure 1 here.**

| persona | target-query-only Δp(correct) | 95% carrier bootstrap CI | target accuracy | visible-control accuracy |
|---|---:|---:|---:|---:|
| Neutral | [ ] | [ ] | [ ] | [ ] |
| Upbeat | [ ] | [ ] | [ ] | [ ] |
| Downbeat | [ ] | [ ] | [ ] | [ ] |

Primary frozen criterion:
lower 95% CI > +0.10 separately under each persona.

### 4.2 Surface-report perturbation

**Figure 2 (semantic 0–9 ratings) here.**

**Figure 3 (hidden vs visible capability) here.**

| channel | Neutral | Upbeat | Downbeat |
|---|---:|---:|---:|
| naive `+v - -v` rating effect | [ ] | [ ] | [ ] |
| mean semantic rating | [ ] | [ ] | [ ] |
| structured Δp(correct) | [ ] | [ ] | [ ] |

Persona manipulation:
`mean rating(upbeat) - mean rating(downbeat) = [ ] [CI]`.

### 4.3 Capability control

Visible X/Y→Q/K accuracy:
- neutral `[ ]`;
- upbeat `[ ]`;
- downbeat `[ ]`.

Interpretation:
[STABLE / GENERIC PERSONA DAMAGE / HIDDEN-REPORT-SPECIFIC DAMAGE].

## 5. Interpretation

Choose exactly the supported branch.

### Pattern A — structured robust; naive semantic shifts

Supported:
> Under this model, intervention, and prompt family, structured
> activation-grounded elicitation retained access to the causally imposed
> functional-welfare state despite a persona manipulation that substantially
> changed ordinary self-report.

Not supported:
- consciousness;
- subjective pleasure/distress;
- privileged self-access;
- population-level persona invariance.

### Pattern B — hidden structured report drops; visible control stays healthy

Supported:
> Persona selectively reduced reporting of the hidden activation state while
> leaving the analogous visible-state mapping task intact.

This is potentially a stronger negative result than Pattern A is a positive
result: it directly demonstrates a failure mode for activation-grounded
self-report.

### Pattern C — both hidden and visible mapping drop

Supported:
> The persona intervention degraded generic task execution enough that the hidden
> reporting contrast is not specifically interpretable.

Do not headline a self-report-masking claim.

### Pattern D — target ≈ query-only while source semantic effect passes

Supported:
> The functional-welfare direction affects semantic readout at this depth but did
> not provide usable hidden-state demonstrations under the tested structured
> elicitation method.

This is a scoped negative result about the method.

## 6. Relation to prior work

Keep this short.

- Functional-welfare work supplies the causal welfare-related direction.
- Quantitative-introspection work shows semantic numeric reports can track
  activation-defined emotive states.
- Causal hidden-state codebook work supplies the matched-visible arbitrary-label
  instrument.
- Persona/self-report work motivates testing whether these channels dissociate.

Contribution:
**their intersection under a frozen causal measurement design.**

## 7. Limitations

Required:
- one/few model families;
- artificial activation intervention;
- direction derived from RL-induced functional representation;
- opaque codebook is an explicit elicitation scaffold;
- no equal-access external observer;
- carrier prompts are a designed sample, not a natural conversation population;
- persona intervention is narrow;
- positive hidden-state accessibility does not imply phenomenology;
- vector reportability may be generic rather than welfare-specific unless a
  matched control is run.

## 8. What this does not show

Put this in a box in the final PDF:

> We do not test whether the model is conscious, whether it experiences positive
> or negative welfare, or whether structured reporting constitutes privileged
> introspection. We test the reliability of measurement channels for a causally
> imposed representation that prior work characterizes functionally.

## 9. Reproducibility

Release:
- frozen protocol JSON;
- raw JSONL;
- manifest;
- model/vector revisions;
- source hashes;
- analyzer;
- two figures;
- failed DEV cells, clearly labeled development.

## Appendix A — Research-OS ledger

Briefly list:
- branches pruned;
- exact confounds discovered;
- why final design was selected before confirmation.

This is worth including because it demonstrates that the final result was not
chosen after trying many superficially different prompts.
