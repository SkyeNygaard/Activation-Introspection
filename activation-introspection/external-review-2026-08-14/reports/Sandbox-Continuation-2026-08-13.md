# Activation Introspection — sandbox continuation

**Date:** 2026-08-13  
**Status:** post-hoc secondary analysis of already-run artifacts. No new Qwen forward pass was possible in this runtime. Any new endpoint below must be labeled exploratory until prospectively rerun.

## Executive result

The strongest new secondary endpoint is a **latent-XOR quartet**. Four rows have a **byte-identical visible prompt**. Two hidden demonstration configurations encode opposite latent sign→label conventions, crossed with two hidden query signs. A quartet counts only if **all four** outputs are correct.

This has a structural success rate of **0** for a deterministic strategy that sees only the visible prompt, sees only the hidden query, sees only the hidden demonstrations, uses a constant label, or uses a fixed sign→token mapping. It therefore tests the joint use of hidden demonstration state and hidden query state in one matched-visible unit.

### Replicates across two independent codebook banks

| bank | target | random | shuffled | query-only/test-only | clean |
|---|---:|---:|---:|---:|---:|
| original test bank | **0.778** | 0.111 | 0.097 | 0.000 | 0.000 |
| confirmation bank | **0.688** | 0.083 | 0.160 | 0.000 | 0.000 |

The target advantage is positive for **all 8 concepts in both banks**. This is a stricter restatement of the causal codebook result, not a preregistered new discovery.

## Training changes arbitrary latent-code learning

On remap-training v2, strength 0.5:

| arm | target XOR quartet | random XOR quartet |
|---|---:|---:|
| base | 0.365 | **0.000** |
| fixed-trained | **1.000** | **0.823** |
| remap-trained | **1.000** | **0.667** |

Seed values for fixed/random are **0.927, 0.771, 0.771**; remap/random **0.760, 0.615, 0.625**.

This is stronger than saying training improves row accuracy. To solve a random quartet, the model must use an arbitrary hidden axis in the demonstrations to determine the latent convention and use the query hidden state to apply it. The fixed-trained adapter succeeds even though it never trained on the reversed convention.

Fixed training beats remap training on random quartet success in all three seeds and in 18/24 seed×direction cells (4 ties, 2 losses), with a mean direction-level advantage of 0.156. Fixed and remap still rank axis difficulty very similarly. The current evidence therefore suggests **reader strength and convention flexibility are separable**: fixed-convention training may strengthen the hidden-state reader while ordinary ICL retains the ability to remap labels. Three seeds are insufficient for a population claim.

At weaker target edits, the same metric gives:

- strength 0.25: base 0.010; fixed **0.990**; remap **0.962**;
- strength 0.15: base 0.000; fixed **0.604**; remap **0.476**.

## The generalization kernel is narrow

The same strict metric localizes the held-out semantic failure:

| held-out arm | model XOR quartet | cheap centroid reader XOR quartet |
|---|---:|---:|
| same exemplar | 0.417 | **1.000** |
| held-out semantic exemplar | **0.014 (1/72)** | **0.972 (70/72)** |
| held-out scrambled | 0.000 | 0.333 |
| held-out random | 0.000 | 0.000 |
| query only | 0.000 | 0.000 |

Thus the model can learn a code over an **identical hidden axis** but almost completely fails when the query is a new vector from the same semantic category, despite the category relation being almost perfectly available to an external four-shot reader.

This is the cleanest current synthesis:

> **The demonstrated interface supports exact-axis latent binding, and training greatly expands which exact axes are bindable. It does not yet show semantic cross-axis abstraction.**

The prior paired semantic-vs-scrambled analysis remains decisive: 9 semantic-only cells vs 8 scrambled-only, exact McNemar p=1.0. Do not revive “semantic introspection” from the one successful quartet.

## Why separate query-twin and mapping-flip metrics were insufficient

- Query twins prove the answer depends on the query hidden state.
- Mapping-flip output sensitivity does **not** alone prove hidden-demo reading, because reversing the mapping also changes visible Q/K demonstration sequences. Clean/no-signal rows show strong visible sequence heuristics.
- The hidden-demonstration twin fixes this by holding visible prompt and query hidden edit constant while changing only hidden demonstration states.
- The four-row XOR quartet requires both hidden dimensions simultaneously.

The quartet should be prospectively frozen in the next run.

## Elicitation does not rescue semantic abstraction

Re-scoring `heldout_elicitation_v1` with XOR quartets gives held-out-semantic success of only:

- baseline 0.028
- eliminate 0.000
- generalize 0.028
- introspect 0.083
- two-groups 0.083

while same-exemplar quartets are 0.47–0.64. Prompt wording can improve or damage exact-axis use, but no tested prompt creates robust held-out semantic relation use.

## Free-form reporting channel is weaker than the original T1 number suggests

The original text-only LLM reader scored 7/24 = 0.292, but it predicts `hammer` on 14/24 reports. Two other successes contain explicit semantic leakage (`eagle`, `island-like`).

New leave-one-carrier-out classifiers trained on the reports from the other two carriers score:

- word TF-IDF logistic: **0/24**
- char TF-IDF logistic: **0/24**
- word/char SVM, centroid, or 1-NN: **0–1/24**

Earlier unsupervised same-concept report similarity was also null. These are post-hoc lexical diagnostics, but they argue against a stable carrier-generalizable verbal code for concept identity. The forced-choice channel can contain information even when the free-form prose does not expose it legibly.

## Retained-trace representation: correction and factorization

A prior sandbox interpretation proposed that separately constructed layer-local concept vectors might be actively mapped toward a common semantic endpoint downstream. That mechanism is **rejected**.

Cross-depth concept identity is already **1.000 at the injection sites themselves** for target vectors at 0.5B, 1.5B and 3B, while random/shuffled controls are near chance. Downstream processing therefore preserves an already cross-layer-aligned semantic geometry; it does not create it.

The final state nevertheless factorizes content and provenance cleanly. For target traces:

- only ~1.5–3.7% of concept-effect energy lies in the injection-depth factor subspace;
- removing the depth subspace leaves cross-depth/cross-carrier concept decoding at **1.000** for all three model sizes;
- removing concept and carrier factor subspaces makes depth provenance easier to decode, reaching ~0.73 / 1.00 / 0.90 in the 0.5B / 1.5B / 3B datasets.

Concept and intervention-depth information are therefore largely **superposed in separable directions**, rather than collapsed into a single canonical state. The eight concept contrasts themselves are high-dimensional (effective rank ≈6.6–6.9 of a maximum 7).

## Reader-depth clue: interesting, not promoted

In `reader_depth_v1`, the fixed four-shot source-axis reader is perfect through layer 32. At layer 35 it is correct on only ~0.545 of rows where the model answers correctly, but ~0.984 of rows where the model answers incorrectly; the inverse association survives mapping/query-sign stratification.

This may indicate that successful downstream processing transforms the marker-state relation away from the original source-axis reader, while failed episodes leave it intact. But it appears abruptly at the final block and the reader and model use different read positions, so this remains a **post-hoc mechanism clue**, not a claim.

## Updated Research-OS frontier

### PROMOTE as secondary evidence

1. **Matched-visible latent ICL is real.** Target XOR quartet replicates on two independent concept banks, with clean/test-only controls at zero.
2. **Codebook training broadens latent binding to arbitrary axes.** Fixed-trained random quartet 0.823; remap-trained 0.667; base 0.
3. **The held-out-semantic failure is now more sharply localized.** Same-axis relation use exists; different-exemplar same-category relation use almost vanishes while a cheap reader remains near-perfect.

### LIVE root questions

1. Does **zero-demonstration** activation-report training itself induce arbitrary latent ICL, or was that capability specific to training on codebook demonstrations?
2. Is cross-axis generalization governed by raw geometry (cosine), semantic relation, or exact axis identity?
3. Can the same saved checkpoint reverse a strong native Q/K polarity when hidden demonstrations demand the opposite mapping?
4. Does training preserve real no-signal abstention when `UNKNOWN` is available?

### PRUNE / demote

- “training destroys semantic selectivity” from random codebook performance;
- “random codebook = false positives”;
- active downstream semantic canonicalization;
- more one-line prompt-conflict descendants;
- free-form prose as evidence of a reliable concept-report channel in the current 24-row study.

## Highest-EVI next run

Train **one zero-demonstration reporter**, save and hash the adapter, and evaluate every interface on that same checkpoint. The draft protocol in `next_latent_binding_protocol_draft_v1.json` freezes:

1. original zero-demo target/random/shuffled/clean;
2. same-axis latent XOR on target/random/shuffled;
3. cross-axis cosine bins;
4. held-out semantic vs geometry-matched scrambled;
5. same-checkpoint native-polarity reversal;
6. a true no-edit query with an explicit `UNKNOWN` answer.

The most informative first gate is simple: **if zero-demo training does not improve arbitrary random latent-XOR quartets, then the meta-decoder interpretation was specific to codebook training and should be narrowed immediately.**

## Runtime boundary

This sandbox has PyTorch/scikit-learn/scipy and all checked-in raw rows/tensors, but no `transformers`, `peft`, cached Qwen weights, or saved LoRA adapters. I exhausted additional inference available from the saved artifacts; a genuinely prospective model result requires executing the saved-checkpoint protocol in the original model environment.
