# Result artifact status

## Trained zero-demonstration reporter, and the loss bug it caught

`report_training_protocol_v2.json` (SHA-256 `6812dc8f…e461d`) freezes the
repaired run; `report_training_protocol_v1.json` (`acb92e81…d8082`) is retained
as the disclosed precursor. Each raw artifact holds 504 rows across an untrained
base arm, a trained held-out-bank arm, and a trained own-bank arm.

V1 reached 0.917 twin-pair accuracy on eight concept directions it never trained
on — and was not a verbalization. Its trained arm held ~5e-9 total probability on
the two labels and a 0.000 full-vocabulary format rate, because the training loss
was a two-way softmax over the label logits that constrained their ordering and
nothing else. V2 changes only the loss to full-vocabulary cross-entropy. Format
and label mass return to 1.000 and held-out accuracy falls to **0.583**, against
0.000 for the untrained base model and 0.208–0.250 for magnitude-matched
controls. All four V2 gates pass; under the current analyzer V1 fails the
verbalization gate.

Note that the pair-wise null is 0.000 for a prompt-only strategy and 0.250 for a
coin flip, not 0.500 — the frozen 0.500 threshold is conservative. Full protocol,
limits, and a disclosed docstring error in the frozen source are in
[`../notes/07-trained-activation-reporter.md`](../notes/07-trained-activation-reporter.md).
Regenerate with `make report-training-report`.

## Stage 1b head screen: pre-registered STOP, and a cross-concept replication

`attention_head_screen_dev_protocol_v3.json` (SHA-256 `759c0850…25856d`) froze
the 64-component universe, the gate algebra, and the stop rule before any model
output was seen. The 72-row raw artifact holds 5,112 scored forwards (SHA-256
`9833a9bf…54bde`); `attention_head_screen_dev_v3_summary.json` is produced by the
hash-locked analyzer, which evaluates the gates itself rather than leaving the
verdict to prose.

**The screen stopped the single-route study.** Two gates failed:

- `query_marker@23` removed 16.2% of the aggregate baseline margin against a
  20% all-head replication threshold. The other three parents passed
  (`final_answer@26` 71.6%, `final_answer@31` 37.1%, `query_marker@21` 29.4%).
- Six individual components cleared the 10% component threshold, against a
  frozen `sparse_go` window of 2–4. Influence is broader than a compact route.

The six passers are `final_answer@26/head-{0,5,10}` (41.5%/36.8%/24.8%),
`final_answer@31/head-7` (27.6%), `query_marker@21/head-2` (18.0%), and
`query_marker@23/head-2` (11.5%). All are 6/6 strata positive with format 1.000
and label-mass retention 1.0000. The layer-26 heads alone exceed 110% of their
own parent's 71.6%, which is redundancy, not decomposition; the protocol forbids
reading these additively.

The same artifact carries an unplanned positive. Its baseline arm is a clean
replication of the causal-codebook effect on three concepts (`bread`, `volcano`,
`violin`) and two carriers that appear in neither the V2 confirmation bank nor
the Stage 1a selection: target accuracy **0.958** against **0.500** for the
exactly query-matched `test_only` arm, mean signed-margin gap 5.19, 6/6 strata
positive, format 1.000. That is a cross-concept generalization check, not a
confirmatory rerun, and it was not the reason the run was commissioned.

Regenerate with `make head-screen-report`.

## DEV-only attention localization (selection, not confirmation)

`attention_localization_dev_protocol_v2.json` (SHA-256 `27c8af5f…e41427`)
freezes the one-concept/one-carrier layer-role screen. The 12-row raw artifact
contains 1,248 patches / 1,284 scored forwards (SHA-256 `530f4f55…d5c1c`);
the fail-closed summary selects query-marker L21/L23 and final-answer L26/L31.
Independent reconstruction found no discrepancy. This is development selection,
not a circuit, QK, or safety result.

## Current evidence: causal opaque-codebook ICL

`codebook_icl_confirm_protocol_v2.json` freezes the repaired model, revision,
layer, strength, literal stimuli, exact design, generation-source hashes,
analysis rule, sampling units, and gates. Its SHA-256 is `fbba4892…ffc39`.

`codebook_icl_confirm_v2_raw.jsonl` contains 576 episode rows and all five
condition scores (2,880 forwards). The paired manifest pins raw SHA-256
`f45d2ac5…7cf20`, model revision `aa8e725…04d1`, protocol/config/source hashes,
environment, directions, tokens, and row count.
`codebook_icl_confirm_v2_summary.json` is regenerated from the raw file by the
checksummed analyzer, which fails closed on hashes, episode/prompt/token/score
invariants, and the exact 8 concept × 3 carrier × 24 cell design before
enumerating all 64,350 crossed-bootstrap resample pairs.

The V2 target accuracy is 0.891 [0.816, 0.995], target minus query-only is +0.391
[0.316, 0.495], target minus the strongest random/shuffled direction is +0.231
[0.137, 0.286], and next-token label-format integrity is 1.000. All frozen gates
pass. The full interpretation and limits are in
[`../notes/06-causal-codebook-icl.md`](../notes/06-causal-codebook-icl.md).

`codebook_icl_test_*` is the inspected V1 precursor. It remains for provenance,
but V2 supersedes it for claims after repairing per-position normalization,
DEV-only centering, transitive source locking, row validation, and exact rather
than Monte Carlo intervals.

## Earlier current evidence: the retained-trace study

`retained_test_qwen05b_raw.jsonl` and `retained_test_qwen05b_summary.json` are
**not** legacy artifacts. They are the confirmatory run described in
[`../notes/05-retained-trace.md`](../notes/05-retained-trace.md): one row per
trial, with the summary carrying the raw file's SHA-256, the model revision, the
git commit, the prompt/carrier hashes, and the device/dtype. Every table and
figure is regenerated from the raw rows by `scripts/analyze_retained.py` and
`scripts/plot_retained.py`; nothing is hand-edited.

`retained_test_qwen05b_raw.acts.pt` holds the captured residual streams used for
the storage probe. It is ~18 MB and is **deliberately ignored** by the
`results/**/*.pt` rule rather than committed. It is a regenerable intermediate:
rerunning `make retained-test` reproduces it bit-for-bit from the pinned model
revision and seed, and the storage numbers follow from it. If a release needs
the storage figures to be verifiable without a GPU, attach this file to the
release rather than adding it to the tree.

## Legacy artifacts

Every result file in this directory other than the causal-codebook artifacts and
the retained-trace files identified above predates the 2026-08-01 audit and is a
**legacy exploratory artifact**. In particular:

- the IFT JSON files were generated with active evaluation dropout, repeated
  prompt pseudo-units, fixed concept→digit mappings, and invalid cross-layer
  vector reuse;
- the transfer/layer profile and IFT series used mismatched injection sites and
  cannot support the former negative correlation;
- the ladder/sweep intervals treat repeated cells as independent and use an
  asymmetric observer;
- `reach_output_qwen05b.json` is a local, ignored, provenance-incomplete audit
  aggregate. Preserve it as history, but do not stage or cite it.

New matched profiles must be generated by `scripts/run_reach_output.py`. Commit
the schema-2 summary together with both checksummed JSONL artifacts under
`results/raw/`; the ignore policy permits those specific raw files. A new IFT
result remains exploratory until its runner also saves item-level records and
multiple independent adapter runs.

The authoritative statuses are in
[`../notes/04-claim-audit.md`](../notes/04-claim-audit.md).

## The repaired-control rerun (2026-08-05)

`retained_test_qwen05b_v2_*` repeats the confirmatory run after
`concepts.random_control` was fixed to seed per concept. Before the fix every
concept received a byte-identical `random` edit, so that arm's 0.125 was
arithmetic rather than a measurement.

Cite `_v2_`. The original `retained_test_qwen05b_*` is retained for provenance
and is not wrong. Its target arm is bit-identical to the rerun's at every layer,
because the control seeding never touched the target arm. What changed is that
the `random` arm's eight concept states now differ by L2 54.5 instead of 0.21,
which is the difference between fp16 noise and a control. The headline did not
move when the control started working, which is the only reason the rerun was
worth doing.

`retained_test_qwen{15b,3b}_*` still carry the degenerate `random` arm. Quote
`shuffled` for those until their reruns land with the recalibrated strength.
