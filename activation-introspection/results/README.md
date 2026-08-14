# Result artifact status

## The "an injection happened" direction: a passed gate, not a claim

`displacement_direction_pilot_qwen05b_v2.json` and
`displacement_direction_qwen3_4b.json` are the gate for
[notes/38](../notes/38-identity-or-displacement.md), which asks whether a trained
reporter reads *which* concept was injected or only *that* the residual stream was
disturbed. Before that can be tested by removing the disturbance, the disturbance
has to be a coherent thing to remove. These two runs check that.

Inject at a quarter depth, read the final block, strength 1.0, eight concepts
crossed with target/random/shuffled. The pooled direction is fitted on development
concepts **and** development carriers, then scored on held-out concepts and three
carriers never used in the fit.

| | Qwen2.5-0.5B | Qwen3-4B |
|---|---:|---:|
| held-out injected vs clean, AUROC | 1.000 | 1.000 |
| share of displacement energy along the mean delta | 0.217 | **0.546** |
| share along the leading component | 0.217 | 0.547 |

**The direction is real and concentrated.** Perfect ordering of held-out injected
states above held-out clean ones on both models, and the mean delta *is* the
leading component to three decimals — so it is one axis, not a mixture that
happens to average out.

**The share is the bound on everything downstream.** It is how much of the
disturbance a rank-1 ablation actually removes. At 0.217 the planned ablation
would have left roughly four fifths of the effect in place and "the reports
survived" would have meant nothing; at 0.546 it means something. That is why the
0.5B pilot was worth running before any training compute was spent.

`displacement_ablation_qwen3_4b.json` answers what the share could not: whether
removing that direction takes the disturbance with it and leaves identity behind.
Both scored on held-out concepts and held-out carriers, with the post-ablation
direction refitted on development states rather than the rows it scores.

| held-out test | before | after |
|---|---:|---:|
| tell injected from clean (AUROC) | 1.000 | **0.500** |
| tell which concept (chance 0.25) | 1.000 | **1.000** |

**Chance on one, untouched on the other.** At this readout the fact of an
injection and the identity of what was injected are linearly separable: one
direction carries all of the first and none of the second.

An earlier version of that table read 1.000 after ablation, because the refit was
fitted and scored on the same 39 states — and 39 points in 2560 dimensions are
almost always separable, so it measured dimensionality rather than signal. The
held-out refit gives 0.500. Recorded because the design was nearly abandoned on
the artifact.

**Not a claim about introspection.** No reporter has been ablated yet, and the
0.5B-versus-4B comparison confounds size with model generation (Qwen2.5 against
Qwen3). Identity was at ceiling before ablation, so a small loss could not have
been detected. One layer, one strength, three clean states per split.

`displacement_direction_pilot_qwen05b.json` is the first pilot and is
**superseded**: it scored held-out concepts against the same two clean states used
to fit, so its held-out column was not held out on the clean side. Kept because
the error is the reason the carrier split exists.

## Programmatic attention: the lever is coverage and context, not kernels

`head_budget_protocol_v1.json` asks the cheapest question nobody had asked of the
programmatic-attention workstream: **a head that has been deleted costs nothing,
so how fast does GPT-2 get if you just delete them?** That is a hard ceiling on
any program, however well written. Three arms — unmodified, k heads deleted per
layer, k heads replaced by the exact lowering of the released first-token program
fused into one projection in and one out — across 7 coverages, 5 lengths, 2
devices, 3,150 paired timing blocks. Both gates pass.

**The exact lowering captures 96–100% of the deletion ceiling on the processor
and 90–99% on the graphics chip.** The project brief's reading of the earlier
1.089× — *"the algebra was never the bottleneck; partial-head projection and
dispatch are"* — is therefore wrong, and this is the fused implementation that
brief proposed as the fix. The earlier number was small because **one head of
twelve is a small share of the work**: three heads buys 1.19×, ten buys 2.09×.
Break-even for the 1.25× the earlier study missed is **34% coverage** at 1024
tokens, and unreachable at any coverage at 64 or 128 tokens.

Three of my five pre-run predictions were wrong, all understating the ceiling,
because I estimated it by counting multiplications and the explicit attention
implementation does not spend its time there. Scope:
[`../notes/27-how-much-can-a-free-head-possibly-buy.md`](../notes/27-how-much-can-a-free-head-possibly-buy.md).

## Half the apparent gain belongs to the baseline, not the program

`head_budget_axes_protocol_v1.json` flips the three choices the study above fixed
arbitrarily: it drops GPT-2's vocabulary projection from the timed region, runs
against the standard fast attention implementation as well as the explicit one,
and goes to 4096 tokens. **Quote `head_budget_axes_v2_*`, not `_v1_*`** — the
first execution failed its own `ceiling_bounds_program` gate, having been
contaminated by other work running on the machine during its first four cells.
The v1 artifacts are kept with that gate recorded false. `head_budget_axes_short_*`
holds the matched-length rerun used to isolate the vocabulary-projection axis.

**The headline: against the attention implementation people actually deploy,
roughly half the advantage disappears** — 46% of it at high coverage and 1024
tokens, 54% at 4096. The pitch for programmatic attention is that it avoids
building the big square score matrix, but the standard fast implementation
already does exactly that, so measuring against the explicit one — as both the
earlier benchmark and the study above did — compares against something nobody
runs. Removing the vocabulary projection moves the other way and is worth up to
26 points. Context length is the one axis where the idea is winning and had not
flattened at 4096.

Under the most realistic conditions available here, at the 25% coverage the
source literature uses, the exact lowering delivers **1.06× at 1024 tokens and
1.13× at 4096** — against a reported ~16% average perplexity cost for that same
coverage. Scope:
[`../notes/28-three-ways-the-ceiling-could-be-wrong.md`](../notes/28-three-ways-the-ceiling-could-be-wrong.md).

## Introspection training loses to a difference-of-means probe

`trained_vs_probe_protocol_v2.json` fits readers on the trained reporter's own
training bank and scores them on its own evaluation bank — eight held-out concept
directions, three held-out carriers, 24 twin pairs. Both fair readers reach
**1.0000** twin-pair; the oracle upper bound adds nothing because there is nothing
left to add; the shuffled-label control sits at 0.5208 row accuracy inside the
frozen band, with twin-pair 0.125, *below* the 0.25 coin-flip null.

The adapter's published four-seed figure is **0.927, range [0.833, 1.000]**. It is
therefore **beaten by 0.073**, with its best seed only tying. At this setup, LoRA
training on activation reports yields a reader strictly worse than the probe one
would have to fit anyway to generate the supervision.

The protocol argued before the run that the outcome was open, on the grounds that
a linear reader might not transfer to unseen directions. It transferred
perfectly, and that reasoning is retracted in the note. The mechanism implied —
a shared "this state was pushed" axis rather than concept-specific content — is
labelled **inference, not measurement**, and the cheap check that would settle it
is named and unrun. Full scope:
[`../notes/12-training-versus-a-probe.md`](../notes/12-training-versus-a-probe.md).

## The privileged-access criterion: the model is dominated by a cheap reader

`matched_reader_protocol_v3.json` tests the confirmed 0.891 against the field's
operative definition of introspection ([arXiv 2508.14802](https://arxiv.org/abs/2508.14802)):
more reliable than a process of equal or lower cost available to a third party.
On the identical 576 frozen episodes, the same forward pass that scores the model
captures the five post-injection residual states, and a four-shot nearest-centroid
reader is fitted on the four demonstrations and asked for the fifth.

The model reproduces its frozen confirmation at **0.8924** (gate: 0.891 ± 0.05).
The reader scores **1.0000**, on both Euclidean and cosine variants, on all eight
concepts. The shuffled-label control collapses to **0.5017**. The paired counts
are the result: **514 both correct, 62 reader-only, and 0 episodes where the model
is right and the reader is wrong.**

`reader_depth_protocol_v1.json` then attacked that result's own stated first
limitation — that the reader reads at the block the injection edits. Same
episodes, same forward pass, reader refitted after **every** block. Blocks 0–8
give exactly **0.5000** (validity control: nothing is edited yet), blocks **9
through 32 give exactly 1.0000**, block 33 gives 0.9427, and only blocks 34–35
fall below the model, to 0.8854 and 0.5920. The frozen verdict fired as
`dominance_is_a_read_site_artifact` because the *final* block was named the
primary statistic; that choice was wrong and is corrected in the note, since by
the last blocks the marker positions are producing their own next-token
predictions rather than storing the concept. The finding is broader than the test
intended: the signal is perfectly decodable across **25 consecutive blocks, ~70%
of depth**, while the model answers at 0.892 throughout.

This does not invalidate the 0.891 — an input-only learner is still pinned at
exactly 0.500 by construction, which is the point that design was built to make.
It bounds what the number may be called: the model reads its own state *worse*
than an outsider does at any of 25 read depths, so no introspection or
privileged-access language is licensed. Two disclosed precursor protocols carry implementation
defects that produced no usable result, and a wrong expectation inside the frozen
v3 protocol is corrected in the note. Full scope and limits:
[`../notes/11-matched-cost-reader.md`](../notes/11-matched-cost-reader.md).

## Certified-donor natural-state report: an uninterpretable null

`natural_report_l27_protocol_v1.json` certified each of twelve fresh arithmetic
pairs individually — both transplants must carry the ordinary answer — and ran the
24-cell reporter on the first five certified pairs in frozen bank order. **Nine of
twelve certified**, against 9.7 expected at the measured per-transplant rate, so
the certification design worked. The reporter then ran for the first time in this
family: natural accuracy **0.500**, query-only **0.500**, difference **0.000**,
twin-pair **0.000**, with format 1.000, label mass 1.000 and sham reproducing
clean to 0.000.

It is **not** a reporting null. The frozen interpretation gate — visible capability
≥ 0.75 — returned **0.533**: the model cannot induce the parity rule with the
arithmetic written out and nothing patched at all. The blocker is the hidden
class, not the site, the transplant, or the interface. See
[`../notes/10-output-ready-arithmetic.md`](../notes/10-output-ready-arithmetic.md).

## Output-ready natural-state pilot: stopped before reporting, on a perfect task

`natural_state_arith_smoke_protocol_v1.json` froze the successor to the route
pilot below. It transplants the residual at the last pre-answer token of a
single-digit arithmetic problem the model solves itself, screens three
prospectively named anchor layers (9, 21, 26) on a development bank, and would
have run the 24-cell episode-remapped `Q/K` reporter on a disjoint held-out bank
at the earliest passing layer.

No anchor passed. Clean answers were **10/10 at conditional probability 1.000**
— the task defect that also afflicted the route pilot is gone — and exact
self-patching reproduced the full logit vector with maximum error **0.0** at
every layer, but bidirectional cross-patching controlled the answer in **0/5**
tasks at all three, with mean normalized recovery +0.001, −0.003 and +0.100.
Replacing the state at the position that produces the answer does not make the
answer follow. No reporting row ran, and this is again an instrument result
rather than evidence about introspective reporting.

The artifact is named `smoke` and is still the citable run: `--smoke` only
truncates the reporter, which never executed.

`natural_state_arith_site_diagnostic_v1.json` is a **post-hoc, development-bank
only** all-layer localization, produced by `scripts/diagnose_answer_site.py`. It
carries no reporting claim and its selected layer is not a pre-registered choice.
It explains the stop: through block 26 the clean pre-answer state does not favour
its own answer over its twin's better than chance under a logit lens, and the
twins differ by under a quarter of the residual norm; from **block 27** the same
transplant makes the donor's digit the full-vocabulary argmax in **10/10**
transplants at mean recovery 0.787. The three anchors were one block short of the
site.

`natural_state_arith_l27_smoke_protocol_v1.json` then froze block 27 as a
disclosed post-hoc site and changed nothing else. It reproduced the development
result exactly — 5/5 tasks, recovery 0.787 — and the **held-out bank did not
confirm: 3/5 tasks** against a frozen 4/5, at recovery 0.697 with clean answers
10/10. The reporter did not run, and the protocol forbids reselecting a layer.
Eight of ten held-out transplants worked; both failures were the same direction.
Pooled over both banks the per-transplant rate is 0.90, at which the 5-task
"4/5 in both directions" criterion fails about a quarter of the time on its own,
so the design's bank was too small for the gate it carried. Full scope, both
stops, and the unused visible capability control are in
[`../notes/10-output-ready-arithmetic.md`](../notes/10-output-ready-arithmetic.md).

## Natural-state route pilot: stopped before reporting

`natural_state_smoke_protocol_v2.json` froze an inference-only feasibility test:
transplant a naturally computed two-hop route state, require it to change the
ordinary answer, then reuse the episode-remapped `Q/K` reporter. The first gate
failed. Clean route answers were 8/10 and cross-patching the layer-9 marker state
worked bidirectionally in 0/5 worlds. Exact self-patching reproduced the full
logit vector with maximum error 0.0, so this is a no-reach result for the tested
site rather than broken replacement plumbing. No reporting rows ran, and this is
not evidence against natural-state reporting. Full scope and a disclosed
diagnostic-metric bug are in
[`../notes/09-natural-state-pilot.md`](../notes/09-natural-state-pilot.md).

## Sensitivity/specificity trade-off in introspection training

`remap_training_protocol_v2.json` (SHA-256 `f29b479d…be0f0`) freezes the citable
run: three training seeds, 4,608 rows each, all four gates passing. Two adapters
are trained on byte-identical episode formats differing only in whether the label
convention is held fixed or re-randomised, then scored against the untrained
model on eight concept directions and two carriers neither saw.

**Training extends the detection floor.** At injection strength 0.15 the
untrained model is at exactly chance (0.500 row, 0.010 twin pairs — blind). Both
adapters, trained only at strength 0.5, read it at 0.790–0.863. At 0.25 the base
is at 0.526 and the adapters at 0.990–0.997.

**Training collapses the advantage concept-derived directions held.** Untrained,
random magnitude-matched directions sit at 0.513 while concept directions reach
0.745. After training, random directions reach 0.913–0.955. **Corrected
2026-08-14** — this read "training destroys semantic selectivity", which treats the
random rows as no-signal trials. They are not: the random direction is injected
into the demonstrations as well as the query and has a correct label, so the
finding is about what can be bound, not about false alarms.

Two nulls are identities of the design rather than measured controls: a
prompt-only learner scores 0.000 on query-twin pairs, and a fixed sign-to-token
probe scores 0.000 on mapping-flip pairs. Both adapters reach 1.000 on the latter,
which settles the probe objection to `report_training`.

`remap_training_protocol_v1.json` and `remap_training_v1_seed{0,1}_*` are the
falsified precursor: v1 predicted fixed-convention training would damage
in-context remapping, and two seeds refuted it. Their summary carries
`gate set v1` and `all_gates_pass=false`, because gate sets are keyed to the
protocol that produced the artifact rather than to the current analyzer.

Full protocol, limits and disclosed deviations are in
[`../notes/08-sensitivity-specificity-tradeoff.md`](../notes/08-sensitivity-specificity-tradeoff.md).
Regenerate with `make remap-training-report`.

## Trained zero-demonstration reporter, and two errors it caught

`report_training_protocol_v3.json` freezes the citable run: four training seeds
named before any of them ran, 504 rows each, seed declared as the inference unit.
V1 and V2 are retained as disclosed precursors.

Across four seeds a LoRA trained on eight concept directions names the sign of a
causally injected hidden state on eight directions and three carriers it never
saw at **0.927 mean twin-pair accuracy, range [0.833, 1.000]**, with format and
label mass at 1.000. The untrained base model is at 0.000 on every seed;
magnitude-matched random and shuffled directions sit at 0.260 and 0.208. Every
seed passes every gate, and every seed beats its own strongest control by
+0.542 to +0.708.

Two precursors, both instructive:

- **V1 scored 0.917 and was not a verbalization.** Its training loss was a
  two-way softmax over the label logits, which fixes their ordering and leaves
  the rest of the vocabulary free; the adapter suppressed both labels to ~5e-9
  total probability and never emitted one. No forced-choice metric can see this.
- **V2 repaired the loss and scored 0.583 — and that number is not the effect.**
  V1 and V2 seeded only the example order, never the adapter initialization, so
  neither could be distinguished from initialization luck. V2 was one
  uncontrolled draw and it fell below all four seeded runs.

The V2 write-up's claim that the 0.917 → 0.583 drop measured the broken loss's
inflation was wrong; the repair cost essentially nothing in accuracy (0.917
degenerate against 0.927 genuine) and only made the output real.

The pair-wise null is 0.000 for a prompt-only strategy and 0.250 for a coin
flip, not 0.500 — the frozen 0.500 threshold is conservative. Full protocol,
limits, and a disclosed docstring error in the frozen source are in
[`../notes/07-trained-activation-reporter.md`](../notes/07-trained-activation-reporter.md).
Regenerate with `make report-training-seeds-report`.

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
