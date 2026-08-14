# Activation-Introspection: deep sandbox pass — 2026-08-13

## Scope

This pass uses only checked-in raw JSONL, saved retained-state activation tensors, and repository source/docs extracted from the supplied archives. It does **not** use repository analyzers for the core new metrics. No Qwen forward pass or LoRA training was possible in this runtime: the archives contain no saved adapters/base weights, and `transformers`/`peft` are absent.

Everything below that was not frozen prospectively in the repository is explicitly **post-hoc secondary analysis**. The purpose is to sharpen or kill mechanisms and choose the next prospective experiment, not to manufacture a new headline from recycled data.

## Executive result

The cleanest synthesis is now **two different reporting regimes**, which later notes accidentally conflated:

1. **Zero-demonstration trained reporter:** a fixed, strongly concept-direction-preferring readout. Training also increases sensitivity to arbitrary axes, but it does not erase target preference.
2. **Demonstrated-codebook trained reporter:** an adaptive same-axis binder. Given demonstrations on an arbitrary hidden axis, it can infer that episode's axis→Q/K convention, including both opposite label mappings.

That makes the later “random = false positive/noise” story untenable for the remap experiments. Those random trials are valid hidden-codebook tasks. High confidence there means the model is **correctly decoding the arbitrary demonstrated axis**, not confidently hallucinating concept X.

This distinction was partly understood earlier in note 13 and then lost again in notes 29–31. The new sandbox analyses make that regression measurable and identify the exact missing factorial experiment.

---

## 1. New correction: the zero-demo random control needs an orientation-invariant sensitivity metric

`report_training_v3` evaluates a trained reporter with no demonstrations. For target concept directions, `+v` has a semantic orientation inherited from concept-minus-filler construction. For a sampled random direction, however, the orientation of `v` versus `-v` is arbitrary.

The published twin metric treats `+random -> Q` and `-random -> K` as the convention. That is fine for **convention accuracy**, but it is not a sufficient measure of **specificity/sensitivity**: a reporter that strongly flips on a random axis with the opposite polarity scores zero even though it is maximally sensitive to that axis.

Rescoring by whether the prediction flips across the byte-identical +/- twin gives:

| condition | base flip rate | trained flip rate |
|---|---:|---:|
| target | 0.042 | **0.927** |
| random | 0.167 | **0.552** |
| shuffled | 0.125 | **0.479** |

Published trained convention twin accuracy was 0.927 / 0.260 / 0.208. So the old control metric hides roughly half of the trained reporter's arbitrary-axis responsiveness.

This **does not** mean selectivity vanished. Using the raw Q−K response slope, target response magnitude is still about 2.8–3.5× the strongest control in every seed. The corrected statement is:

> Training strongly preferentially sensitizes the model to held-out concept-derived directions, **and also broadens sensitivity to arbitrary matched directions**. It does not make the zero-demo reporter equally sensitive to everything.

This is a materially different claim from either “perfect semantic specificity” or “training destroys semantic selectivity.”

### A common detector geometry emerges across independent LoRA seeds

For each of 16 arbitrary control axes (8 random + 8 shuffled), I recovered the raw Q−K response slope and subtracted the base model's slope. Across the four independently trained LoRA seeds:

- mean pairwise Pearson correlation of the **training-induced** 16-axis response profile: **0.743**;
- rank-1 energy fraction of the 4×16 induced-response matrix: **0.802**;
- 100,000-label-shuffle permutation test: **p ≈ 1e-5**.

Pairwise seed correlations are 0.598–0.844. The common profile is not inherited from the base model: base-vs-trained-mean correlation is −0.169 on random axes.

Interpretation, cautiously: independent LoRAs converge toward a low-dimensional fixed response geometry over arbitrary axes. This is a useful mechanistic hypothesis, not a prospective result.

The response profile is also highly carrier-stable within most seeds (random-axis direction-profile correlations across carriers are generally ~0.84–0.96), arguing against a carrier-specific artifact.

---

## 2. The demonstrated random arm is evidence for adaptive binding, not false positives

The source confirms that in `remap_training_v2`, the **same random direction is injected into all four demonstrations and the query**. The demonstrations establish how its sign maps to Q/K. Therefore the random arm has a correct answer.

The strongest statistic is the existing mapping-flip pair: hold the hidden axis/order/query sign fixed and require the model to be correct under **both opposite Q/K conventions**.

At strength 0.5:

| arm | target mapping-flip | random mapping-flip |
|---|---:|---:|
| base | 0.677 | 0.438 |
| fixed-label trained | **1.000** | **0.939** |
| remap-trained | **1.000** | **0.889** |

A fixed Q/K detector cannot produce that random-axis mapping-flip performance. Even the adapter trained with a fixed label convention rebinds the arbitrary hidden axis to the episode's demonstrated convention.

This basic insight was already recognized in note 13, then later epistemic bookkeeping regressed: notes 29–31 recast the same correct random-axis decoding as “false positives,” “noise,” and “meaningless directions.” The task does not license that interpretation.

### Confidence analysis reverses the later story

On the random-codebook arm, absolute Q/K margin predicts **correct decoding** very well after training:

- fixed adapters: AUROC ≈ **0.79–0.89** by seed;
- remap adapters: AUROC ≈ **0.79–0.92**;
- base: ≈ **0.49**.

Keeping only the highest-confidence random rows pushes trained random-codebook accuracy toward 1.0. Thus the observation in notes 29–31 is real but its meaning is opposite: confidence enriches for **successful arbitrary hidden-axis decoding**, not false alarms.

### Same geometric difficulty across two training recipes

Pooling over the three seeds, per-direction random-codebook accuracy for fixed-trained versus remap-trained adapters correlates:

- Pearson **r = 0.965**, p ≈ 0.00010;
- Spearman **ρ = 0.874**, p ≈ 0.0045.

So different training recipes share which arbitrary hidden axes are easier/harder to bind.

### But demonstrations do more than exploit a fixed native detector

I joined the same eight random axes across the zero-demo and demonstrated paradigms. Zero-demo absolute response magnitude does **not** predict demonstrated decoding accuracy (Pearson near 0; Spearman non-significant). The most vivid case is `banana`: all four zero-demo trained reporters have a strong **negative** native slope (about −3.2), yet with demonstrations that axis is decoded at ~0.917 fixed-trained / 0.847 remap-trained across mappings.

Because these are different trained adapters, this is not a causal factorial result. But it is exactly why the next experiment should put no-demo and demonstrated evaluation on the **same saved checkpoint**.

---

## 3. Held-out semantic result: no semantic advantage survives an exact paired test

The headline comparison remains:

- held-out semantic: 12/144 twin successes = 0.083;
- held-out scrambled: 11/144 = 0.076.

Matching cells exactly by category pair, draw, carrier, and episode cell:

- semantic-only successes: **9**;
- scrambled-only: **8**;
- both: 3;
- neither: 124;
- exact McNemar/binomial p = **1.0**.

So there is no evidence whatsoever that real semantic grouping helps relative to the matched arbitrary grouping.

The previously tempting diagnostic—12 semantic prediction flips and all 12 in the correct direction—is real, but post-hoc. Scrambled has nearly the same success count, and the semantic-vs-scrambled paired test kills the semantic interpretation.

A nuance worth retaining: semantic and scrambled each beat the random/query-only floor in post-hoc paired comparisons (p roughly 0.01–0.02). That suggests weak **generic hidden-state relational sensitivity**, but the successes are strongly exemplar/category-pair concentrated, so I would not promote it without a prospective replication. The safe statement is:

> The model shows weak hidden-state sensitivity under held-out exemplars, but **no semantic advantage over matched arbitrary grouping**. The external reader's large semantic advantage remains absent in the model.

This is more precise than “the model never sees anything” and much weaker than semantic abstraction.

---

## 4. Retained-state geometry: identity survives drift; active canonicalization is falsified

I used the saved activation tensors directly and classified injected concept identity at the **final residual layer**, holding out both injection depth and carrier from the centroid used for that test.

Using repaired controls:

| artifact | target concept ID | random | shuffled | chance |
|---|---:|---:|---:|---:|
| Qwen 0.5B held-out test, 6 depths | **1.000** | 0.215 | 0.146 | 0.125 |
| Qwen 1.5B repaired dev, 3 depths | **1.000** | 0.236 | 0.069 | 0.125 |
| Qwen 3B repaired dev, 3 depths | **1.000** | 0.194 | 0.083 | 0.125 |

This is a clean geometric restatement of the retained-trace phenomenon: concept directions are built separately at each injection layer, yet their propagated final states preserve concept identity across depth far better than independent random/shuffled controls.

### Kill test: does the network actively converge layer-local concept states?

No. For every pair of injection depths, I compared same-concept cosine at the moment the later intervention is inserted versus at the final layer.

Mean same-concept cosine:

| artifact | at later injection layer | final | change |
|---|---:|---:|---:|
| 0.5B repaired test | 0.806 | 0.722 | **−0.084** |
| 1.5B repaired dev | 0.670 | 0.479 | **−0.191** |
| 3B repaired dev | 0.610 | 0.383 | **−0.227** |

**Every target depth pair decreased.** So the attractive “the network canonicalizes different layer-local concept vectors into one shared endpoint” mechanism is falsified by the saved tensors.

What survives is subtler:

> Concept identity is preserved under substantial representational drift.

The final state remembers **what** was injected even though the exact concept vectors from different injection depths grow less aligned downstream.

### The final state also retains provenance

Injection depth remains decodable above chance even when holding out both concept and carrier. This is not target-specific—random/shuffled controls also retain depth information—so it looks like a generic intervention/provenance signature rather than semantic content. The representation is therefore not a single canonical concept state; it carries content plus generic history/provenance.

For the repaired 0.5B test, a simple balanced variance decomposition at the final layer attributes about **53%** of target-state variance to concept identity versus ~11% for random/shuffled controls. On repaired 1.5B/3B dev tensors, the target concept main effect remains larger (about 41% / 35%) than controls (~20–23%). Treat these as descriptive because the decomposition is post-hoc.

---

## 5. Independent control/provenance checks

### Old 1.5B/3B random controls are unmistakably degenerate

This was already disclosed in the repository. The sandbox independently verifies it from activation tensors without trusting the analyzer:

- concept-centered random-state norm is only about **0.03–0.04%** of target concept variation in the old 1.5B/3B artifacts;
- i.e. the eight “random concept” states differ only at fp16-noise scale.

The repaired 0.5B random control has ~94% of target's concept-centered norm.

### Old shuffled controls preserve target inter-concept geometry

In old 1.5B/3B artifacts, target-vs-shuffled concept Gram correlation is essentially **1.000 at every injection depth**. This is the fingerprint of the old shared RNG stream applying one common coordinate permutation to every concept vector.

This does **not** make shuffled an arithmetic identity; unlike the old random arm, it still varies by concept. It means the old shuffled control is a different kind of control: it preserves inter-concept relational geometry while moving it into the wrong model coordinate basis. Current per-concept shuffling destroys that shared geometry. Cross-size comparisons should state that difference explicitly.

### The target depth shape survives a repaired independent dev bank

At the three overlapping injection depths in 1.5B and 3B, old held-out-test target accuracy and later repaired-dev target accuracy correlate:

- Pearson **r = 0.994** across six model×depth cells;
- Spearman **ρ = 0.943**.

This is exploratory and does not retroactively repair the old test controls, but it strongly suggests the early→late target depth profile itself was not caused by the broken random constructor. Deep-layer nulls also survive repaired controls.

---

## 6. Protocol bookkeeping defects that still need explicit repair

### Frozen report-training protocols still state the wrong pair identity

`report_training_protocol_v1.json`, `v2`, and `v3` all literally say that byte-identical opposite-label twins imply a prompt-only strategy “scores exactly **0.500 on pairs**.” The current `run_report_training.py` still contains the same sentence.

That is false for the reported pair statistic “both twins correct”: a deterministic prompt-only model must emit the same answer on both byte-identical visible prompts and therefore scores **0.000** pair accuracy. The summaries/CLAIMS later corrected this, so the repository currently contains a source/protocol-versus-analysis contradiction.

The raw result is not harmed; the protocol documents should be superseded explicitly rather than silently treated as correct preregistrations.

A grep shows related stale 0.25/0.500 pair-null language remains in HANDOFF, notes 12/23/24/25, `trained_vs_probe_protocol_v2`, and the associated runner/manifest.

### Positive handoff check: clustering replication was handled correctly

I independently checked the second 14-rule clustering artifact because it looked like an easy-to-miss failed replication. The handoff **already records it correctly as “DID NOT REPLICATE, do not quote.”** This is one place where the epistemic handoff worked as intended.

---

## 7. CoT branch: quantified instrument contamination

The generated reasoning branch was already demoted because generation broke the anchor. The raw text explains why very concretely:

- **575/576 = 99.8%** of non-forced generations contain `query/key/queries/keys` semantics;
- **481/576 = 83.5%** contain explicit visible pattern/sequence/alternation language.

The first traces literally infer that Q/K “alternate” from the visible demonstrations. Thus the generated-CoT instrument is almost universally hijacked by label semantics and visible pattern search. This is not just a few cherry-picked examples.

If generation is revisited, use semantically inert single-token labels and freeze the parser before running anything.

---

## 8. The current `none` arm is better, but still not the final specificity test

The unrun remap-v3 `none` arm correctly removes the query edit while preserving demonstrations. That is much better than treating random as a false-positive arm.

But a binary Q/K margin on a no-answer trial is confounded by standing label preference. Existing clean/no-edit rows show this can be large: one trained seed predicts K on all clean rows with mean |Q−K| margin ~3.25; another predicts Q on all rows with ~2.67. Strong binary margin with no signal is therefore not itself a calibrated false positive.

A better protocol has an explicit `UNKNOWN`/`NEITHER` response or a threshold frozen from separate clean development data, plus a **cross-axis** condition: demonstrations use axis A, query receives matched axis B. That separates same-axis relational binding from generic perturbation detection.

---

## 9. What Research OS should promote now

### PROMOTE

1. **Causal hidden-state codebook access exists.**
2. **Zero-demo training preferentially amplifies held-out concept-derived axes, while also increasing arbitrary-axis sensitivity.** Use orientation-invariant control sensitivity in addition to convention accuracy.
3. **With demonstrations, trained reporters can adaptively bind arbitrary hidden axes to episode-specific label conventions.** Mapping-flip performance is the clean evidence. Do not call correct random-axis decoding a false positive.
4. **Held-out semantic abstraction remains unsupported.** Semantic and scrambled are indistinguishable pair-by-pair.
5. **Retained concept identity survives downstream representational drift.** Active convergence/canonicalization is not supported.

### PRUNE / RETRACT

- “random directions in remap are false positives/noise/confidently wrong”;
- “confidence enriches false alarms” in notes 29–31;
- active canonicalization of layer-local concept directions;
- any semantic reading of the 0.083 held-out result;
- 0.25 as the deterministic prompt-only twin null.

### LIVE ROOT QUESTION

**Does zero-demo activation-report training itself create the adaptive demonstrated-axis binder?**

The current evidence is cross-experiment because the zero-demo and remap reporters are different adapters. One saved checkpoint can settle it.

The draft protocol in `next_meta_decoder_protocol_draft_v0.json` crosses:

- base vs trained;
- target/random/shuffled;
- no demonstrations;
- same-axis demonstrations under both opposite Q/K mappings;
- demonstrations on axis A with query axis B;
- no-edit query;
- explicit `UNKNOWN`;
- and the frozen note-23 semantic-vs-scrambled held-out test on that exact checkpoint.

The critical falsifier is **native-polarity override**: identify arbitrary axes that push the trained reporter strongly toward Q or K with no demonstrations, then demonstrate the opposite mapping. If the same checkpoint reliably follows the demonstrations, that is strong evidence for adaptive hidden-axis binding rather than a fixed detector. If it cannot, the meta-decoder synthesis is wrong.

This should outrank cross-layer transfer and any further one-line prompt descendants.

---

## 10. Sandbox execution boundary

I searched both supplied archives for model/adaptor files (`*.safetensors`, `adapter_config.json`, model bins/checkpoints): none are present. The runtime has PyTorch but no `transformers`/`peft`, and no cached Qwen checkpoint was found.

I attempted the repository tests twice:

1. vanilla `pytest` fails collection because the package is not installed;
2. with `PYTHONPATH=src`, collection reaches the package but all relevant modules import `transformers`, which is absent.

So this sandbox cannot honestly create a new Qwen/LoRA forward-pass result. The offline scripts here compile and run successfully, and all headline secondary metrics above are regenerated from raw artifacts/tensors.

## Bottom line

The project is stronger after this pass, but the strongest story changed again:

> **Training does not simply create a noisy “something moved” detector. With no demonstrations it produces a concept-preferring but broadened fixed response geometry; with demonstrations, trained reporters can flexibly bind even arbitrary hidden axes to an episode-specific codebook. The unresolved question is whether those two behaviors are the same learned capability.**

The next training run should be designed to answer exactly that on one saved checkpoint, while simultaneously resolving true no-signal specificity and post-training semantic generalization.
