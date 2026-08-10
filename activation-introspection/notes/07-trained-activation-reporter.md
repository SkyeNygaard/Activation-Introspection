# A trained zero-demonstration activation reporter

Run date: **2026-08-10**

## Question

The frozen causal-codebook study showed that four in-context demonstrations can
teach an episode-specific mapping from a causally injected hidden state to an
opaque label. This asks the training version, which is the SPAR Introspection
Training project's actual subject: can a LoRA fitted on one bank of concept
directions report the sign of a causally injected hidden state on directions and
carriers it never saw, with no demonstrations in context?

Everything else in this repository is a zero-weight-update intervention. This is
the first result here that trains anything.

## Why chance is exactly 0.500, and why that is not a measured control

Both members of a query twin are byte-identical in visible text and have
opposite correct labels. Any strategy that reads only the prompt must give both
members the same answer, so exactly one of the two is right. The estimand is
therefore the **twin pair** — a cell counts only when both members get their
opposite labels right — and any prompt-only learner scores 0.000 on pairs and
0.500 per row.

This is why no input-only arm is run. Its value is an identity, not a
measurement. The [reality-check objection](https://arxiv.org/abs/2605.26242) to
introspection benchmarks, that an input-only classifier can match hidden-state
prediction, cannot apply to a design where there is no input to classify on.

## Design

`Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, MPS,
bfloat16 base weights with float32 adapter parameters. One signed edit of the
concept direction at layer 9, normalized strength 1.0, at the marker token — the
same site and strength as the frozen in-context work.

Three banks, mutually disjoint:

| bank | concepts | role |
|---|---|---|
| train | guitar, harbor, lantern, meadow, satellite, teapot, tunnel, whale | LoRA training only |
| eval | garden, camera, train, banana, eagle, library, hammer, island | never trained; the V2 confirmation bank behind the 0.891 in-context result |
| centering | ocean, bread, volcano, violin, spider, hospital, desert, clock | estimates the center only |

Centering is estimated on a bank that is neither trained nor evaluated, so no
test direction depends on another test direction. That defect was found and
fixed in the in-context V2 protocol and the same rule applies here.

Six training carriers; three evaluation carriers withheld from training, one of
them deliberately terse (`Nothing changed.`) and one deliberately long. LoRA rank
16, alpha 32, learning rate 1e-4, six epochs, one seed.

Arms at evaluation: `target` (the concept direction), `random` and `shuffled`
(magnitude-matched controls), and `clean` (no edit, which has no correct label
and measures standing label bias). Each is scored on the untrained base model,
the trained model against the held-out bank, and the trained model against its
own training bank.

## V1: the gates passed and the result was still not a verbalization

V1 reached **0.917** twin-pair accuracy on the held-out bank — 22 of 24 cells,
7 of 8 concepts positive — against 0.000 for the untrained base model and
0.208 for the strongest control. All three preregistered gates passed.

It should not be reported as introspection, and the reason is in the same
artifact:

| arm | twin-pair accuracy | full-vocabulary format rate | mean label mass |
|---|---:|---:|---:|
| base, target | 0.000 | 1.000 | 0.994 |
| trained, target | **0.917** | **0.000** | **~5e-9** |
| trained, random | 0.167 | 0.000 | ~0 |
| trained, shuffled | 0.208 | 0.000 | ~0 |
| trained on its own bank | 1.000 | 0.000 | ~0 |

The trained model puts essentially no probability on `Q` or `K`. Its actual next
token is never a label. The 0.917 is a two-way forced choice between tokens the
model would not emit — a discriminative readout, not a report.

The cause is a one-line design error in the training loss. V1 minimized
cross-entropy over the two label logits alone. That constrains their *ordering*
and nothing else, so the optimizer was free to suppress both labels against the
rest of the vocabulary while keeping the correct one on top. The in-context
result never had this failure mode because nothing was trained: its
unrestricted next-token format integrity was 1.000.

This is worth stating plainly because it generalizes past this experiment.
**Restricting an introspection-training loss to the answer options produces a
probe wearing the model's output head, and every accuracy metric computed on
those options will look excellent while the model has stopped answering.** A
forced-choice evaluation cannot detect it. Only an unrestricted
full-vocabulary check can.

V1 is retained as a disclosed precursor at `results/report_training_v1_*`.

## V2: the repair

V2 changes exactly one thing: the loss is full-vocabulary cross-entropy on the
correct label token. Model, layer, strength, banks, carriers, optimizer settings,
seed, and the three original gates are unchanged. A fourth gate is added and is
evaluated by the analyzer rather than asserted in prose: the trained arm must
retain at least 0.90 full-vocabulary format rate and at least 0.50 mean label
mass, which is the check V1 failed.

All four gates pass. 504 rows, raw SHA-256 `a3d6361e…db6c68`.

| arm | condition | twin-pair | per-row | margin | label mass | format |
|---|---|---:|---:|---:|---:|---:|
| untrained base | target | 0.000 | 0.479 | 0.14 | 0.994 | 1.000 |
| trained | random | 0.250 | 0.542 | 0.00 | 0.996 | 1.000 |
| trained | shuffled | 0.208 | 0.479 | 0.05 | 0.995 | 1.000 |
| **trained** | **target** | **0.583** | 0.729 | 3.30 | **1.000** | **1.000** |
| trained, own training bank | target | 1.000 | 1.000 | 10.74 | 1.000 | 1.000 |

Fixing the loss restored verbalization completely — format 1.000 and label mass
1.000, against V1's 0.000 and 5e-9 — and **cut held-out accuracy from 0.917 to
0.583**. That drop is the measurement of how much V1's number was inflated by a
readout the model would never have spoken.

What survives is still a genuine effect, and the floor it clears is lower than
0.500. A prompt-only strategy scores 0.000 on pairs, and a row-independent coin
flip scores 0.250; the preregistered 0.500 threshold is conservative rather than
the null. Against that, the trained reporter reaches 0.583 on directions it never
saw and carriers it never saw, while the untrained model is at 0.000 and the
magnitude-matched controls are at 0.208–0.250. The readout is therefore
direction-specific and not a generic "something was perturbed" detector — the
in-context study found random and shuffled directions learnable at 0.658/0.660,
and training did not reproduce that.

Two things temper it. Generalization is partial and uneven: the adapter is
perfect on its own eight training directions and loses 41.7 points on the eight
held-out ones, and per-concept accuracy ranges from 1.000 (`garden`, `eagle`,
`hammer`) to 0.000 (`library`, `train`), with 5 of 8 concepts above the
threshold. And the `clean` arm answers `K` on 100% of rows in every arm,
including the base model, so there is a strong standing label bias that the twin
structure controls for but does not remove.

## Disclosed deviations

- A two-concept smoke was run and inspected before the full V1 run. It scored
  1/2 pairs on the held-out bank, 2/2 on the training bank, and 0/2 for the base
  model. No setting, gate, or stopping rule changed afterwards.
- V1's outcome was inspected before V2 was frozen. V2 is therefore a disclosed
  repair-confirmation after an inspected precursor, not a pristine first look.
  The repair is to the loss function, which V1's own saved label-mass column
  identifies as defective independently of any accuracy number.
- **`src/introspect/report_training.py` contains a wrong statement in its module
  docstring**, and it is still there. It says a prompt-only strategy scores
  0.500 on twin pairs. The correct values are 0.500 per row and 0.000 on pairs.
  Nothing computes from that sentence — the gate threshold is a separate frozen
  constant and the code is unaffected — but the file's hash is locked into both
  frozen protocols, so correcting it would invalidate the V1 and V2 artifacts for
  the sake of a comment. It is disclosed here and stated correctly in the
  analyzer and in this note instead. Fix it in the next protocol version.
- The verbalization gate was added to the analyzer after V1 was inspected. Under
  the current analyzer V1 reports `gates_pass=false`; under the analyzer that
  existed when V1 ran, it reported true. Both summaries in `results/` are
  produced by the current analyzer, so V1's recorded verdict is the corrected
  one.

## What does not follow

- One model, one layer, one strength, one binary variable, **one training seed**.
  A single seed cannot separate the effect from initialization luck; the design
  calls for independent seeds and they have not been run.
- The mapping is fixed and global, not re-randomized per episode. With zero
  demonstrations an episode-specific mapping is unanswerable in principle, so
  this design gives up the property that makes the in-context result strong.
  Trained and in-context results are therefore **not** directly comparable, and
  the 0.891 figure is not a baseline for this number.
- Strength 1.0 adds a direction with the mean residual norm at the edited
  position. This is a large, out-of-distribution edit, not a naturally occurring
  fluctuation.
- Reporting the sign of a planted edit is not verbalizing a naturally occurring
  internal computation, and it is not privileged self-access: no equal-cost
  external observer is tested.
- The control directions are magnitude-matched, not damage-matched.
- No independent human review or different-hardware reproduction has occurred.

## Artifacts

- V1 precursor: `results/report_training_protocol_v1.json`,
  `results/report_training_v1_raw.jsonl` and its manifest and summary;
- V2: `results/report_training_protocol_v2.json`,
  `results/report_training_v2_raw.jsonl` and its manifest and summary;
- runner and analyzer: `scripts/run_report_training.py`,
  `scripts/analyze_report_training.py`;
- regenerate the summary with `make report-training-report`.
