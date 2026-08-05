# Experiment plan

> **Superseded by `04-claim-audit.md` (2026-08-01).** This v1 preregistration is
> preserved to show what was specified before the pilot. Its IID bootstrap,
> observer-headline, sample-size, novelty, and clean-negative assumptions failed
> the later audit and must not be used for a new run.

Written before results exist. Amend by appending dated entries, not by editing
predictions after the fact.

## Sequence

| # | Experiment | Question | Gate to proceed |
|---|---|---|---|
| 0 | `make smoke` | Does anything work? | Concept cosines near 0; injection visibly changes output; greedy determinism restored after hooks |
| 1 | Strength × layer sweep, behaviour only | Where does injection change output without destroying it? | A usable window exists: KL from clean is elevated but text stays coherent |
| 2 | Detection AUROC | Can the model tell *something* was injected? | AUROC > 0.5 on real directions AND ≈ 0.5 on shuffled controls |
| 3 | Identification, forced choice over 8 concepts | Can it say *what*? | Accuracy > 1/8 with the null arm at chance |
| 4 | **Observer arm** | Does a clean model do just as well from the output alone? | — this is the result, not a gate |
| 5 | Pre-verbalization | Does detection survive with no behavioural channel? | — this is the result |
| 6 | Matched-behavioural-effect comparison | Does accuracy differ at equal KL? | — this is the result |

Experiments 0–3 are plumbing. 4–6 are the paper. Do not spend the budget on 0–3.

## Metrics

- **Detection AUROC** over injected vs clean trials. Not hit rate: a YES-biased
  model gets 100% hit rate and 0 bits of information.
- **Identification accuracy**, forced-choice over the concept bank, chance = 1/k.
  Free-form answers get graded separately and generously (synonyms count) — report
  both, since free-form is where confabulation is visible.
- **Behavioural effect size**: mean KL(intervened ‖ clean) over next-token
  distributions on `NEUTRAL_TASK`. This is the x-axis that makes the observer
  comparison fair.
- **Introspector − observer gap**, with a bootstrap CI. The headline number.
- **False alarm rate** on shuffled/random controls.

## Sample sizes

8 concepts × 5 strengths × ~6 layers × 20 trials ≈ 4800 forward passes per model.
At ~0.2 s per short generation on an M4 Pro that is under half an hour for a 1.5B
model. There is no reason to run underpowered here — budget 20 trials minimum per
cell and bootstrap everything.

## Pre-registered conditions table

For every reported cell, all six of these must be run, not a subset:

1. clean (no injection)
2. concept direction
3. shuffled control, matched norm
4. random Gaussian control, matched norm
5. concept direction, observer arm
6. shuffled control, observer arm

Arms 3–4 give the false-alarm baseline; 5–6 give the behavioural-inference
baseline. A plot missing arms 5–6 is not interpretable.

## What gets written down as a failure

Failures are results here and belong in the report:

- If identification never exceeds chance at any scale available, that is a
  clean negative bound on small-model introspection, worth stating.
- If identification exceeds chance but the observer arm matches it, that is a
  *positive* finding about confounding in the existing literature.

## Extension: programmatic attention (only if time remains)

Replace one attention head with a hand-written program — previous-token, local
window, or token-matching — and ask two questions: how much task performance is
lost, and can the model detect that one of its heads has been replaced? The
second question is the bridge back to this repo's main line and is, as far as I
can tell, unasked. Keep it as a stretch section, not a second project.

---

## 2026-08-01 audit amendment

The observer does not receive the same transcript or damage state, repeated
trials are not independent, and a small-model null is not a clean bound unless
format, reach, training, and damage controls pass. The programmatic-attention
novelty statement was also not supported by a systematic review. All new work
uses the v2 protocol in `04-claim-audit.md`.
