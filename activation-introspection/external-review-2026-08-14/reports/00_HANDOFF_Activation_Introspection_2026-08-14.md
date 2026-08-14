# Activation Introspection Research Handoff

**Date:** 2026-08-14  
**Project:** SkyeNygaard/Activation-Introspection  
**Primary research target:** SPAR — Introspection Training / Self-Knowledge / Faithfulness  
**Status:** Existing artifacts deeply audited and reanalyzed. No new Qwen forward pass was possible in the sandbox because model weights / adapters / `transformers` / `peft` were unavailable. The strongest new results below are post-hoc secondary analyses of already-run raw artifacts and should be prospectively confirmed before being treated as standalone publication claims.

## 1. Executive summary

The underlying experiments are substantially stronger than some of the current prose, but several interpretations in the repo are stale or wrong.

The cleanest current synthesis is:

> **The demonstrated interface supports causal hidden-state binding to an exact activation axis. Training greatly expands which exact axes are bindable, including arbitrary directions. However, the tested model shows almost no semantic cross-axis abstraction: when the query uses a different exemplar from the same semantic category, performance collapses despite the relation being almost perfectly readable by a cheap external reader.**

A new post-hoc endpoint, the **latent-XOR quartet**, gives the strongest causal version of the codebook result. Four rows share a byte-identical visible prompt. Hidden demonstration states define one of two opposite sign→label conventions, crossed with two hidden query signs. Quartet success requires all four outputs correct. Deterministic visible-only, query-only-hidden, demonstration-only-hidden, constant-label, or fixed sign→token strategies have structural quartet success 0.

This endpoint reproduces across two independent codebook banks and sharply separates exact-axis latent binding from semantic cross-axis generalization.

## 2. Claims that currently survive

### A. Causal matched-visible hidden-state use

Latent-XOR quartet accuracy on two independent concept banks:

| bank | target | random | shuffled | query-only / clean |
|---|---:|---:|---:|---:|
| original test bank | **0.778** | 0.111 | 0.097 | 0.000 |
| confirmation bank | **0.688** | 0.083 | 0.160 | 0.000 |

The target advantage is positive for all 8 concepts in both banks.

Interpretation: the model can jointly use hidden information in demonstrations and the hidden query under byte-identical visible prompts. This is controlled state-conditioned computation. Do **not** call it strong introspection or semantic self-knowledge.

### B. Training strongly expands exact-axis latent binding

On remap-training v2 at strength 0.5:

| arm | target XOR | arbitrary-random XOR |
|---|---:|---:|
| base | 0.365 | **0.000** |
| fixed-trained | **1.000** | **0.823** |
| remap-trained | **1.000** | **0.667** |

Fixed-trained random seed values: 0.927 / 0.771 / 0.771.  
Remap-trained random seed values: 0.760 / 0.615 / 0.625.

At weaker target edits:
- strength 0.25: base 0.010; fixed 0.990; remap 0.962
- strength 0.15: base 0.000; fixed 0.604; remap 0.476

The fixed adapter was trained only on target concept-derived directions under one label convention. Yet at evaluation it usually succeeds under both opposite Q/K mappings for arbitrary directions. This argues against a simple learned fixed `+vector -> Q` rule.

### C. Fixed-convention training does not appear to destroy remappability

Fixed-trained adapters outperform remap-trained adapters on arbitrary random quartets in all 3 seeds and in 18/24 seed×direction comparisons (4 ties, 2 losses), with mean direction-level advantage ~0.156.

This comparison is exploratory and only three seeds exist. Best current mechanism hypothesis:

> **Reader strength and label-convention flexibility may be separable: fixed training strengthens hidden-state readout while pretrained ICL machinery remains capable of remapping the output convention.**

Do not promote this to a population claim without a prospective rerun.

### D. Semantic cross-axis abstraction fails under the tested interface

Latent-XOR on held-out semantic exemplars:

| condition | model | cheap centroid reader |
|---|---:|---:|
| same exemplar | 0.417 | **1.000** |
| held-out semantic exemplar | **0.014 (1/72)** | **0.972 (70/72)** |
| scrambled | 0.000 | 0.333 |
| random | 0.000 | 0.000 |
| query only | 0.000 | 0.000 |

Paired semantic-vs-scrambled analysis: 9 semantic-only cells vs 8 scrambled-only; exact McNemar p=1.0. Do not revive the earlier “semantic abstraction” narrative.

Important sample-size caveat: the 72 quartet cells do not represent 72 independent semantic generalization problems. The original design uses only two selected category pairs and eight unique held-out semantic query exemplars, repeated across nuisance structure. The failure is real for the tested cases but should not be generalized broadly.

### E. Prompt elicitation does not rescue semantic abstraction

Held-out-semantic latent-XOR after prompt variants:
- baseline 0.028
- eliminate 0.000
- generalize 0.028
- introspect 0.083
- two-groups 0.083

Same-exemplar remains ~0.47–0.64. Prompting changes exact-axis access but does not produce robust semantic cross-axis generalization.

### F. Free-form verbalization is weak

The original text-only LLM reader scored 7/24 = 0.292, but predicts `hammer` on 14/24 reports and two successes contain explicit semantic leakage (`eagle`, `island-like`).

Leave-one-carrier-out classifiers on report text:
- word TF-IDF logistic: 0/24
- char TF-IDF logistic: 0/24
- other SVM / centroid / 1-NN variants: 0–1/24

Interpretation: the forced-choice latent channel can carry information while free-form prose does not expose a stable carrier-generalizable lexical code for concept identity.

## 3. Major interpretation corrections

### Random is not a false-positive arm

In the remap/codebook experiment, the arbitrary random direction is used in demonstrations and the query. The sign has a valid Q/K answer. High trained random performance means the model can use a demonstrated arbitrary hidden direction as a codebook signal.

It does **not** show:
- concept hallucinations;
- no-signal false positives;
- “did something move?” replacing “is concept X active?”;
- confident wrong semantic reports.

This correction should be propagated through README, HANDOFF, CLAIMS, project briefs, notes 08/29/31, literature boundary, and analyzer comments.

### Twin-pair structural null is 0, not 0.25 or 0.5

For deterministic greedy decoding on byte-identical visible twins requiring opposite labels, a visible-only strategy must output the same label twice, so both-correct twin accuracy is structurally **0**.

0.25 is only an independent fair-coin benchmark. Some frozen protocol text incorrectly says 0.500 for twin-pair accuracy; later analyzers/claims corrected it. Those protocols should be explicitly superseded rather than silently treated as clean preregistrations.

### Earlier same-axis “semantic abstraction” claims should be retired

Previously demonstrated-axis or same-exemplar success can be explained by exact-axis/prototype matching. The held-out exemplar experiment is the relevant abstraction test, and it fails.

## 4. Retained-trace mechanistic findings

The retained-trace result itself broadly survives, but one attractive interpretation was falsified.

### Cross-depth concept identity is preserved

At final residual state, held-out cross-depth/cross-carrier concept identity is near-perfect for target traces and near chance for repaired random/shuffled controls.

However, cross-depth concept identity is already 1.000 at the injection sites themselves. Therefore downstream blocks do **not** create a shared semantic canonical endpoint.

Exact same-concept vector similarity actually decreases downstream. The better description is:

> **The network preserves an already cross-layer-aligned concept geometry through substantial representational drift.**

### Content and intervention provenance are largely separable

At final state:
- only ~1.5–3.7% of concept-effect energy lies in the injection-depth factor subspace;
- removing the depth subspace leaves concept decoding at 1.000 across 0.5B / 1.5B / 3B;
- after removing concept/carrier factors, injection depth remains decodable, suggesting a generic provenance trace.

Do not describe this as full canonicalization.

### Old scale controls have archival defects

The old 1.5B/3B random controls show the fingerprint of the pre-fix shared-control constructor. Old shuffled controls also preserve target inter-concept geometry through a shared coordinate permutation.

This weakens old control interpretation, but later repaired runs strongly reproduce the target depth profile (old-vs-repaired target overlap r ~0.994), so the main early-to-late retained-trace pattern likely remains real.

## 5. Novelty assessment

Confidence in the narrow empirical measurements is high. Confidence in a strong novelty claim is lower.

Closest occupied territory:
- Ji-An et al.: in-context neurofeedback / activation-derived labels, including non-semantic directions.
- Steering Awareness: trained detection of unseen steering vectors with transfer governed by geometric similarity.
- Introspection Fine-Tuning (IFT): training can improve introspective behavior and transfer to another introspection task.
- Activation Oracles / related learned activation readers: broader learned activation decoding and OOD generalization.

Therefore these broad claims are **not** novel by themselves:
- models can report activation-related information;
- training can improve activation reporting;
- geometric similarity constrains transfer;
- external decodability can exceed learned verbalization.

What may be distinctive:
1. the **byte-identical visible-prompt latent-XOR construction**, which jointly requires hidden-demo and hidden-query use;
2. the combination of **arbitrary exact-axis binding** with **near-zero semantic cross-axis abstraction** in one controlled interface;
3. a prospective geometry sweep asking whether semantic relatedness adds anything after controlling cosine similarity;
4. same-checkpoint native-polarity reversal as a direct test of adaptive latent rebinding.

Treat latent-XOR as a promising methodological contribution, but because it was discovered post hoc it needs prospective confirmation.

## 6. Highest-EVI next experiment

Do not continue the long prompt-descendant branch. Use one saved checkpoint to resolve the root question.

### Train

Train **one zero-demonstration reporter** using the report-training recipe. Save the adapter/checkpoint.

The key causal question is whether ordinary zero-demo activation-report training transfers to the demonstrated latent-codebook interface, or whether the arbitrary-axis result only occurs because the existing adapters were themselves trained on codebook episodes.

### Evaluate the same saved checkpoint on

1. **Original zero-demo task** — reproduce training endpoint.
2. **Same-axis latent-XOR** — target / random / shuffled, strengths 1.0 / 0.5 / 0.25 / 0.15.
3. **Cross-axis geometry kernel** — demonstrations use axis u; query axis v at cosine 0.95 / 0.8 / 0.6 / 0.3 / 0.0.
4. **Semantic cross-axis transfer** — frozen held-out semantic design plus geometry-matched scrambled comparison.
5. **Native-polarity override** — measure an axis's zero-demo Q/K slope, then demonstrate the opposite convention on that same axis.
6. **True no-signal specificity** — no query edit and explicit `UNKNOWN` / `NEITHER`, not forced Q/K margin alone.

### Primary kill rule

If zero-demo training **does not** improve arbitrary same-axis latent-XOR, then the current “general latent reader” interpretation is wrong: the arbitrary binding effect is specific to codebook-training exposure.

### Strong positive outcome

A compelling result would be:
- zero-demo training improves arbitrary same-axis XOR;
- anti-native-polarity demonstrations still reverse the checkpoint correctly;
- cross-axis performance decays smoothly with geometry;
- semantic relatedness adds predictive value beyond geometry.

That would move the work from useful preliminary evidence toward a stronger standalone contribution.

## 7. SPAR relevance

### Belinda Li — Introspection Training

Strongest fit. The work asks what introspection training actually teaches. Current evidence suggests a decomposition:

**latent readout -> latent binding -> semantic abstraction -> verbalization**

Training strongly improves the first two; the third remains nearly absent in the tested interface, and free-form verbalization is weak.

### Noah Siegel — Faithfulness / Self-Knowledge / Introspection

The project cleanly distinguishes **private causal access to internal state** from **semantic self-knowledge about what that state means**. A model can look very introspective on an opaque codebook while failing to abstract across semantically related internal representations.

### AI Control / white-box monitoring

A useful safety implication is that held-out prompts are not enough. A monitor may generalize across prompt distributions and output conventions while failing under a different internal realization of the same dangerous cognition. Evaluation should include **held-out internal representations**.

## 8. Epistemic status

### Very high confidence
- raw JSON/JSONL parse and major metrics reproduce;
- no audited raw-SHA mismatches;
- latent-XOR grouping is structurally valid in the analyzed artifacts;
- the two codebook banks reproduce the target-vs-control pattern;
- random-arm “false positive” interpretation is wrong;
- held-out semantic performance is very low relative to the cheap reader in the tested cases.

### High confidence, bounded
- codebook training greatly improves arbitrary exact-axis binding;
- fixed-trained adapters retain counterfactual convention flexibility;
- free-form reporting is weak under simple carrier-generalizable lexical diagnostics.

### Moderate / hypothesis only
- fixed training is better than remap training generally;
- ordinary zero-demo introspection training induces the same adaptive latent reader;
- semantic generalization is absent in general rather than only in the small tested exemplar set;
- latent-XOR is sufficiently novel for standalone publication.

## 9. Files in this handoff

### Start here
- `00_HANDOFF_Activation_Introspection_2026-08-14.md` — this document.
- `Sandbox-Continuation-2026-08-13.md` — latest consolidated post-hoc analysis and frontier.
- `Activation-Introspection-Sandbox-Audit-2026-08-13.md` — deep independent audit of repo, manifests, claims, handoff and literature boundary.

### Reproducibility / strongest secondary result
- `analyze_latent_binding_secondary.py` — independent latent-XOR analyzer.
- `latent_binding_secondary_results.json` — machine-readable latent-XOR results.
- `next_latent_binding_protocol_draft_v1.json` — prospective same-checkpoint protocol; **draft, not frozen, not run**.

### Additional analysis
- `Sandbox-Deep-Pass-2026-08-13.md`
- `sandbox_deep_pass.py`
- `sandbox_deep_pass_results.json`
- `sandbox_extra_pass.py`
- `sandbox_subspace_factorization.py`
- `sandbox_freeform_cv.py`
- `Sandbox-Followup-2026-08-13.md`
- `sandbox_followup_analysis.py`
- `sandbox_followup_results.json`

### Independent original-audit reconstruction
- `Activation-Introspection-independent-audit.py`
- `Activation-Introspection-independent-metrics.json`

### Package / integrity
- `Sandbox-Continuation-Bundle-2026-08-13.zip`
- `Sandbox-Continuation-SHA256SUMS.txt`

Raw source archive status: `Archive(2).zip` (~29 MB) is included in this Drive folder. `Archive(3).zip` (~308 MB) could not be transferred through the Drive connector because the runtime could not register the large local file as a connector file reference after two attempts. It remains available in the original ChatGPT/Library source set. The live public repo `SkyeNygaard/Activation-Introspection` on GitHub should be treated as the canonical code source; request `Archive(3).zip` separately only if the archived raw snapshot is specifically needed.

## 10. Instructions to the next agent

1. Read this handoff, then `Sandbox-Continuation-2026-08-13.md`, then the original sandbox audit.
2. Do **not** reintroduce random=noise/false-positive language.
3. Do **not** use 0.25 or 0.5 as the structural deterministic twin/quartet null.
4. Treat latent-XOR as exploratory until prospectively rerun.
5. Treat the semantic dataset's true independent semantic sample as small despite many nuisance rows.
6. Do not continue note-38 / one-line prompt descendants as the central frontier.
7. If compute is available, run the prospective same-checkpoint latent-binding protocol before opening another sibling branch.
8. Save the adapter/checkpoint and every raw artifact / protocol hash so later comparisons do not require retraining.
9. If preparing SPAR materials now, frame the work as careful empirical preliminary evidence about **what activation-report training learns**, not as a solved theory of introspection.
