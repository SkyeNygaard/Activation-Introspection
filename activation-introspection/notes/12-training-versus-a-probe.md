# Introspection training does not beat reading the state, it loses to it

Run date: **2026-08-11**

## Question

Belinda Li's project description says supervision for verbalization training
"comes cheaply from the internals themselves: probe readouts, feature activation
values, or the measured effects of ablation and patching." That makes one
comparison unavoidable. If a probe fitted on the same states the adapter trained
on does as well as the adapter, introspection training is probe distillation with
extra steps.

[`07`](07-trained-activation-reporter.md) reports the trained reporter at **0.927
mean twin-pair accuracy over four seeds, range [0.833, 1.000]**, on eight concept
directions and three carriers withheld from training. This fits readers on the
adapter's own training bank and scores them on the adapter's own evaluation bank.

## I expected this to be close, and said so before the run

The protocol argues at length that the outcome is open. In
[`11`](11-matched-cost-reader.md) the reader saw four labelled demonstrations of
the *same* direction it was queried on, and won trivially. Here every reader has
to transfer the sign-reading operation to eight directions it has never seen,
from a bank the apparatus explicitly screens for near-collinearity. A linear
reader has no guarantee of transferring at all.

**That reasoning was wrong, and the run says so.**

## Result

| reader | row accuracy | twin-pair | vs adapter 0.927 |
|---|---:|---:|---|
| centroid (difference of means) | 1.0000 | **1.0000** | **+0.073** |
| logistic (L2-regularized) | 1.0000 | **1.0000** | **+0.073** |
| oracle, told the direction | 1.0000 | 1.0000 | upper bound |
| shuffled training labels | 0.5208 | 0.125 | control ✓ |

24 twin pairs, 48 states. The shuffled-label control sits at 0.5208 row accuracy
— inside the frozen ±2 SD band — and its twin-pair falls *below* the 0.25
coin-flip null, which is the signature of a consistent-but-wrong predictor rather
than a noisy one. The control works.

**Verdict by the frozen rule: `training_does_not_exceed_a_probe`.** The simplest
possible probe — a difference of two means — reads held-out directions perfectly,
and the trained adapter does not.

The adapter is not merely matched. It is **beaten by 0.073**, and its best seed
only ties.

## What this means for introspection training

At this setup, LoRA training on activation reports produces a reader that is
strictly worse than the probe you would have to fit anyway to generate the
supervision. There is no version of "the model learned to introspect" that
survives this: the information was linearly available the whole time, the
training data was that same linear signal, and the trained model recovers less of
it than a two-centroid comparison does.

This is the same shape as [`11`](11-matched-cost-reader.md) — the in-context
interface at 0.891 against a matched reader at 1.000 — but it closes a different
door. Note 11 bounded a zero-training interface. This bounds *training*, which is
the intervention Li's project proposes.

## The mechanism, and its epistemic status

**Inference, not measurement.** The probe transfers to unseen directions
perfectly, which it could only do if the eight concept directions share a large
common component — a generic "this state was pushed" axis rather than
concept-specific content. The apparatus permits this: the bank screen admits
off-diagonal cosines up to 0.5, which is a great deal of shared structure.

If that is right, it unifies every result in this repository:

| result | reading under the shared-axis account |
|---|---|
| in-context 0.891, matched reader 1.000 | the model partially reads a generic axis |
| trained 0.927, probe 1.000 | training also reads that axis, slightly worse |
| trained model reads *random* directions at 0.913–0.955 ([`08`](08-sensitivity-specificity-tradeoff.md)) | exactly what reading a generic axis predicts |
| untrained selectivity 0.745 vs 0.513 | the only place concept-specific content shows up |
| linear decodability across 25 blocks ([`11`](11-matched-cost-reader.md)) | a planted axis survives the residual stream almost intact |

That is a coherent account, and it is currently **an inference from one
unexpected generalization result, not a measurement.** The decisive check is
cheap and has not been run: compute the pairwise cosines of the centred eval
directions, and the cosine between the fitted probe weight and each held-out
direction. If the probe weight has a consistent positive component along every
unseen direction, the account is measured. If not, it is wrong and something more
interesting is happening.

Do not cite the unified account until that runs.

## Limits

- One model, one layer, one strength, one training recipe. A different adapter,
  loss, or layer could behave differently, and [`07`](07-trained-activation-reporter.md)
  already documents how much the loss function mattered.
- The comparison is against the adapter's *published* four-seed numbers rather
  than adapters re-run in this process. Those artifacts are frozen and their
  evaluation bank is identical, but the adapters were not re-scored here.
- The readers are scored only on the state-to-sign decision. The adapter also
  emits a formatted token, which it does at format rate 1.000 — a cost, not an
  excuse.
- **The probe's perfect score is itself suspicious in the useful sense**: a task
  a two-centroid classifier solves at 1.000 across held-out directions is an easy
  task, and the right response is a harder one — weaker injections, natural
  states, or multi-way rather than binary content.

## Artifacts

- protocol, raw, manifest, summary: `results/trained_vs_probe_protocol_v2.json`,
  `results/trained_vs_probe_v2_raw.jsonl`,
  `results/trained_vs_probe_v2_raw.manifest.json`,
  `results/trained_vs_probe_v2_summary.json`;
- disclosed precursor: `results/trained_vs_probe_protocol_v1.json` and
  `results/trained_vs_probe_smoke_v1_*`. V1 gated the control on twin-pair
  accuracy at 0.30. That is miscalibrated — the twin-pair null is 0.25 with SD
  0.088 over 24 pairs, so 0.30 sits 0.57 SD above the null and fails about a
  quarter of the time on a healthy control. The six-episode smoke tripped it,
  which prompted re-deriving the null rather than accepting the verdict; the gate
  moved to row accuracy, which has 48 units instead of 24 and a tighter band,
  before any confirmatory run. The smoke also showed both fair readers at 1.000
  on two training concepts, so the direction of the result was visible before the
  confirmatory run and no tuning or stopping decision followed from it;
- runner: `scripts/run_trained_vs_probe.py`.
