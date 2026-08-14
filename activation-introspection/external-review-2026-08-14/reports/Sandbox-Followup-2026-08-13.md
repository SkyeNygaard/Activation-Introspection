# Activation-Introspection — sandbox follow-up

Date: 2026-08-13

## Scope

This follow-up uses only the checked-in raw JSONL and retained `.acts.pt` tensors from `Archive(3).zip` plus the earlier independent audit. It does **not** import the repository analyzers and does not run a model forward pass. All analyses below are therefore exploratory/post-hoc unless they reproduce a previously frozen estimand.

The sandbox has PyTorch/NumPy/SciPy but no `transformers` or `peft`, and neither archive contains base-model weights, tokenizer files, or saved LoRA adapters. New Qwen inference/training is therefore not executable in this environment without bringing in model artifacts.

## 1. Most important new synthesis: training and demonstrations appear to do different things

Two existing experiments become much more informative when put side by side.

### Zero-demonstration trained reporter (`report_training_v3`)

Four training seeds, held-out concept bank:

| condition | trained twin-pair mean | per-seed |
|---|---:|---|
| target concept direction | **0.927** | 1.000 / 0.833 / 1.000 / 0.875 |
| random direction | 0.260 | 0.250 / 0.125 / 0.333 / 0.333 |
| shuffled direction | 0.208 | 0.292 / 0.167 / 0.208 / 0.167 |

Here there are **no in-context demonstrations**. Training is strongly selective for held-out concept-derived axes. It does not make arbitrary random/shuffled directions remotely as reportable as the target bank.

### Trained reporter with a demonstrated hidden-state codebook (`remap_training_v2`, strength 0.5)

| arm | target twin-pair | random twin-pair |
|---|---:|---:|
| base | 0.490 | 0.026 |
| fixed-trained, mean over 3 seeds | **1.000** | **0.910** |
| remap-trained, mean over 3 seeds | **1.000** | **0.826** |

In this task the prompt demonstrates the hidden axis before the query. Once that interface is present, the trained model can use arbitrary random directions extremely well.

### Interpretation

The existing data are inconsistent with the simple claim that training "destroys semantic selectivity" or makes the model an unconditional arbitrary-perturbation detector.

A more interesting hypothesis is:

> **Training installs or strengthens a latent hidden-state codebook interface: without demonstrations it preferentially reports concept-derived axes; with demonstrations it can rapidly bind an arbitrary demonstrated axis to the output convention.**

This is not causally established by the cross-experiment comparison because the two protocols differ in more than demonstration presence. It is, however, a high-value root hypothesis that the current repository has not isolated.

### Highest-value clean experiment

Run one saved trained checkpoint under a factorial evaluation where everything except demonstration presence is held fixed:

- model state: base vs trained;
- context: query-only vs fixed-mapping demonstrations vs remapped demonstrations;
- hidden direction: target vs magnitude/separation-matched random vs shuffled;
- no-signal: no query edit, scored only for abstention/margin unless an explicit `UNKNOWN` label is trained;
- same held-out bank, carriers, layer, strength, output labels, and full-vocabulary scoring across cells.

Primary interaction: does adding demonstrations disproportionately increase random-direction twin accuracy **after training**, while target performance is already high without demonstrations?

That experiment directly distinguishes:

1. concept-specific reporter training;
2. a generic steering/anomaly detector;
3. a trainable meta-decoder that can bind demonstrated hidden axes;
4. generic instruction-following / label-mapping improvement.

## 2. Held-out semantic result should be narrowed, not reversed

Recomputed from `heldout_semantic_v1_raw.jsonl`:

| arm | twin successes / 144 | accuracy | prediction flips inside byte-identical twins |
|---|---:|---:|---:|
| same exemplar | 75 | 0.521 | 75 |
| held-out semantic | **12** | **0.083** | **12** |
| held-out scrambled | 11 | 0.076 | 16 |
| held-out random | 2 | 0.014 | 6 |
| query only | 2 | 0.014 | 4 |

The important post-hoc detail is that **all 12 semantic-arm prediction flips go in the correct direction**; there are zero wrong-way flips. The scrambled arm has 11 correct-way and 5 wrong-way flips.

The effect is also heterogeneous:

- `birds_buildings`: 1 / 72 semantic twins succeed (0.014)
- `body_weather`: 11 / 72 succeed (0.153)

This does **not** rescue semantic abstraction. Overall semantic twin accuracy (0.083) is essentially the same as scrambled (0.076), and the external reader remains near-perfect on the semantic state. The defensible revision is:

> **There is no robust semantic generalization advantage over the matched scrambled arm. There may be weak/brittle hidden-state relational sensitivity in one category pair, but the current experiment does not establish that the model is semantically classifying unseen exemplars.**

So wording such as "the model never sees the category" is slightly stronger than the data require; "the model does not reliably exploit the category relation under this interface" is safer.

The 12/12 directionality observation is post-hoc and shares substantial dependence across prompts/draws; it should generate a future prediction, not a new confirmatory claim.

## 3. Retained-trace tensors contain a strong cross-depth geometry signal

Using the saved activation tensors only, I carrier-centered each concept state, averaged over carriers, and compared concept centroids produced by different injection depths at the final recorded layer.

### Final-layer cross-depth concept matching

| model | arm | same-concept cosine | off-concept cosine | bidirectional top-1 concept match |
|---|---|---:|---:|---:|
| Qwen 0.5B | target | **0.722** | -0.103 | **1.000** |
|  | random | 0.019 | -0.003 | 0.167 |
|  | shuffled | 0.007 | -0.001 | 0.150 |
| Qwen 1.5B | target | **0.630** | -0.090 | **1.000** |
|  | random | 0.001 | ~0 | 0.104 |
|  | shuffled | 0.261 | -0.037 | 0.804 |
| Qwen 3B | target | **0.532** | -0.076 | **0.979** |
|  | random | 0.026 | -0.004 | 0.121 |
|  | shuffled | 0.192 | -0.021 | 0.508 |

The target traces are therefore extremely consistent in **concept identity across injection depth**: a target state created at one injection layer is almost always nearest to the same concept created at another injection layer.

The trajectory is already strong at intermediate readouts. For example, in 0.5B target cross-depth top-1 matching is 1.000 at every reported read layer from 6 through 23. Random controls stay near chance.

### What this does and does not add

This strengthens the existing retained-trace story geometrically: late behavioral inability is not accompanied by loss of concept identity across injection sites. It is consistent with a storage/routing dissociation.

It is **not** a clean new semantic-canonicalization result. The shuffled control itself retains substantial cross-depth concept identity in 1.5B and 3B, showing that deterministic concept-indexed transformations can create part of the same geometry. It also overlaps conceptually with prior work on distributed transformations of steering vectors. Treat this as a mechanism diagnostic/figure, not the project's novelty headline.

A cleaner follow-up would compare these target trajectories to natural-exemplar manifolds or use controls matched not only in norm/separation but also in cross-depth transport geometry.

## 4. What the sandbox can still do without model weights

The existing artifacts support several additional defensible analyses:

1. **Full raw-trace interaction audit.** Put every training/control result into one common schema: base/trained × demo/no-demo × target/random/shuffled, with carrier/concept/seed heterogeneity and full-vocabulary margins. This can quantify which apparent effects survive across interfaces without pretending the protocols form a randomized factorial.
2. **Held-out semantic failure-mode decomposition.** Check whether the 12 successful semantic twins are predictable from exemplar geometry, reader margin, carrier, demonstration order, category pair, or query exemplar. The current concentration in `body_weather` makes this worth doing as a diagnostic.
3. **Retained-trace geometry atlas.** Measure cross-depth nearest-neighbor matching, principal angles, CKA/RSA, and concept-vs-control transport as a function of readout layer. This is cheap on the saved tensors and can reveal whether the trace rotates into a stable subspace before behavioral usability disappears.
4. **Reproducibility/epistemic repair.** Generate a local patch (without pushing) that fixes `random != false positive`, removes the 0.25 structural-null language, narrows the note-23 prose, fixes remap-v3's stale `claim_boundary`, and reconciles the handoff with the new saved-adapter protocol.
5. **Freeze the next experiment.** Produce a protocol JSON + runner patch for the training × demonstrations factorial and the post-training held-out-semantic evaluation, ready for execution on the user's MPS/GPU environment.

## 5. Research-OS frontier after this follow-up

### PROMOTE

- Matched-visible hidden-state codebook access exists.
- Training robustly improves zero-demonstration reporting of held-out concept-derived directions.
- Held-out semantic category generalization is not demonstrated despite highly readable semantic geometry.
- Existing training results are better described as **interface-dependent generalization** than as loss of semantic specificity.

### LIVE — highest EVI

1. **Training × demonstrations interaction:** does training create a meta-decoder for demonstrated hidden axes?
2. **Post-training held-out semantic generalization:** does the same saved checkpoint turn an unseen exemplar/category relation into reportable structure?
3. **True no-signal specificity:** what happens on an unedited query when abstention is actually available?
4. **Privileged-access comparator:** only after the task/interface is stabilized and cost matched.

### PRUNE / demote

- More one-line prompt-conflict descendants.
- "Random directions = false positives."
- "0.25 is the deterministic twin null."
- Generic "training models to detect steering is novel" framing; existing literature already occupies that space.
- Cross-depth vector convergence as a standalone novelty claim.

## 6. Best next external compute run

If compute is available for only one new training checkpoint, save the adapter and use it for **two frozen evaluation suites**:

**Suite A — root mechanism (primary):** base/trained × demo/no-demo × target/random/shuffled/none. This tests the meta-decoder hypothesis.

**Suite B — semantic generalization (secondary, same checkpoint):** rerun the frozen note-23 semantic/scrambled/same-exemplar design after training. Do not tune categories or wording on the checkpoint.

This yields much more information per training run than the current remap-v3 headline. Cross-layer transfer can be tertiary on the same saved adapter.

## Reproducibility files

- `sandbox_followup_analysis.py`: independent offline analysis script.
- `sandbox_followup_results.json`: machine-readable outputs from that script.
