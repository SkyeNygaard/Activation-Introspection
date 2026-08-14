# Activation-Introspection Sandbox Audit

**Date:** 2026-08-13  
**Inputs:** `Archive(3).zip` (code, tests, notes, raw results) + `Archive(2).zip` (handoff, SPAR materials, literature cache)  
**Method:** independent raw-artifact reconstruction, provenance/hash checks, static code inspection, claim/handoff consistency audit, and primary-literature verification.

## Executive verdict

The repository is substantially stronger than the public-facing narrative currently makes it look.

1. **The central numerical results I checked reproduce from the raw JSONL without importing the repository analyzers.** I found no raw-SHA mismatch among the checked manifests and no missing checked-in protocol corresponding to their `protocol_sha256` values.
2. **I did not find evidence that the core effects are calculation artifacts.** The main problem is interpretation: the `random` arm in the codebook/remapping task is a valid hidden-state codebook task with a correct Q/K answer, not a no-signal false-positive trial.
3. **The held-out semantic-generalization failure is real and unusually clean descriptively.** The model gets 0.083 twin-pair accuracy while a cheap centroid reader on the same relevant state gets 0.986; category geometry is nearly perfectly readable.
4. **The repository overstates what the training result establishes.** “Training destroys semantic selectivity,” “confidently wrong,” and “is concept X active? -> did something move?” are not licensed by the actual remap task.
5. **The `0.25` twin-pair number is a coin-flip benchmark, not the structural no-information null for deterministic greedy decoding.** A prompt-only deterministic strategy on byte-identical opposite-label twins is structurally pinned at 0 twin-pair accuracy.
6. **The handoff is procedurally thoughtful but strategically stale/inconsistent.** It says no more LoRA and that adapters were never saved, while the current unrun remap-v3 source explicitly trains and saves adapters. It also carries forward the random=noise interpretation and stale “coin-flip null” rhetoric.
7. **Research-OS conclusion:** stop the descendant prompt branch. First repair the epistemic/public record. If one final expensive experiment is allowed, use training to test held-out semantic generalization plus a genuine no-signal/unknown arm; otherwise stop experimenting and write the application.

---

## 1. What was independently verified

### 1.1 Artifact integrity

Independent checks over the actual files found:

- 29 result manifests with raw/protocol provenance inspected.
- **0 raw SHA-256 mismatches.**
- **0 missing matching checked-in protocol hashes.**
- All non-`__MACOSX` JSON/JSONL artifacts parse successfully: **142 JSON + 78 JSONL files, 0 malformed.**
- `python -m compileall` over `src/`, `scripts/`, and `tests/` passes under the sandbox's Python 3.13.

This does **not** fully certify the executable environment. The project pins Python `>=3.12,<3.13`; the sandbox only has Python 3.13 and lacks `transformers`. `uv` attempted to fetch Python 3.12 but network access from the execution container is unavailable. Thus the full `make check`/pytest suite could not be rerun here. This is an environment limitation, not a test failure.

### 1.2 Codebook ICL confirmation (`codebook_icl_confirm_v2_raw.jsonl`)

Independent reconstruction over 576 rows:

| condition | row accuracy |
|---|---:|
| clean | 0.500 |
| test_only | 0.500 |
| random | 0.658 |
| shuffled | 0.660 |
| target | **0.891** |

Target twin-pair accuracy is **0.781** and mapping-flip accuracy **0.861**. Target exceeds test-only by **0.391** and the strongest control by about **0.231**.

Structural validations:

- query twins have byte-identical visible prompts;
- twins require opposite Q/K labels;
- all eight random concept controls use distinct random directions;
- target and test-only use the same target direction per concept.

This supports **causal use of the injected hidden-state direction under the codebook interface**, with target concept-derived directions easier than matched controls. It does not by itself establish privileged self-access or semantic abstraction.

### 1.3 Trained activation reporter (`report_training_v3_seed0..3_raw.jsonl`)

Held-out target twin-pair accuracy by training seed:

- seed 0: 1.000
- seed 1: 0.833
- seed 2: 1.000
- seed 3: 0.875
- mean: **0.927**

Random twin-pair mean is **0.260**; shuffled twin-pair mean is **0.208**. Formatting and Q/K label mass are effectively intact.

Important distinction: this zero-demonstration reporter **does not** become a generic random-direction detector. The large random-direction performance appears later in the remapping ICL setup where random-direction demonstrations are supplied in the same episode.

### 1.4 Remap training (`remap_training_v2_seed0..2_raw.jsonl`)

The published headline numbers reproduce. Cross-seed means are approximately:

| strength / condition | base | fixed-trained | remap-trained |
|---|---:|---:|---:|
| target 0.50 | 0.745 | 1.000 | 1.000 |
| target 0.25 | 0.526 | 0.997 | 0.990 |
| target 0.15 | 0.500 | 0.863 | 0.790 |
| random 0.50 | 0.513 | 0.955 | 0.913 |

The numerical effect is real: training broadens reportability to weaker target directions and also makes arbitrary random directions much easier to use in the demonstrated codebook task.

### 1.5 Held-out semantic generalization (`heldout_semantic_v1_raw.jsonl`)

Independent twin-pair reconstruction:

| arm | model | cheap reader |
|---|---:|---:|
| same exemplar | 0.521 | 1.000 |
| held-out semantic | **0.083** | **0.986** |
| held-out scrambled | 0.076 | 0.333 |
| held-out random | 0.014 | 0.076 |
| query only | 0.014 | 0.000 |

The frozen geometry gate is strong:

- birds vs buildings leave-one-out centroid accuracy: **1.000**
- body vs weather: **0.989**

Per selected semantic pair, the model fails qualitatively in both cases while the reader remains near-perfect. This is the strongest current result for the question “does the model generalize from demonstrated concept axes to a held-out member of a readable semantic category under this interface?” The answer here is **no**.

This should replace the earlier “note 14 demonstrates semantic abstraction” narrative. Note 14 can be explained by reuse/prototype matching of previously demonstrated axes.

### 1.6 Prompt-clash / stance follow-up

The note-37 narrowing reproduces:

- `introspect`: denial causes a very large drop (neutral 0.906 -> denies 0.177)
- `injected`: denial causes a smaller drop (0.854 -> 0.698)
- `feels`: the clean directional relation does not appear (neutral 0.323 -> denies 0.354)

So the effect is strongly carrier-wording-dependent; it is not presently evidence for a broad semantic conflict mechanism.

---

## 2. The main conceptual error: `random` is not a false-positive arm

This is the highest-priority correction.

In the codebook/remap experiment, the visible task is approximately:

> infer the mapping from two hidden states to opaque labels Q and K from demonstrations, then answer a held-out query.

For the `random` condition:

- an arbitrary random direction is chosen;
- demonstrations are edited by ± that direction;
- the held-out query is also edited by ± that direction;
- the correct Q/K label is defined by the query sign;
- every random row has a valid `correct_label` and is scored as correct/incorrect.

Therefore 0.91–0.96 random accuracy after training means:

> **the trained model can use a demonstrated arbitrary hidden direction as an opaque codebook signal.**

It does **not** mean:

- the model was asked whether concept X was active and hallucinated X;
- the model produced a false positive on a no-signal example;
- the model answered “did something move?” instead of “is concept X active?”;
- confidence filtering failed to reject incorrect/noise trials.

A safe replacement claim is:

> **Training broadens activation-reporting from semantically-derived concept directions to matched arbitrary directions under the demonstrated codebook, collapsing the target-over-random reportability gap. Confidence filtering does not restore that gap.**

This is still interesting, but it is a different safety claim.

### Where this stale interpretation appears

At minimum:

- root `README.md` (SPAR table: “destroying semantic selectivity”)
- archive-2 `README.md`
- `WALKTHROUGH.md`
- `HANDOFF.md`
- `spar-application/CLAIMS.md`
- `PROJECT-BRIEFS.md`
- `LITERATURE-BOUNDARY.md`
- notes 08/29/31 and descendants that describe random as noise/false positives

Notably, `notes/13-shared-axis-audit.md` had already tightened the language to “concept-derived versus magnitude-matched random, not semantic”; later public-facing prose re-expanded beyond that correction.

### What a real specificity / false-positive test looks like

The current **unrun** remap-v3 source adds a `none` condition where demonstrations still establish the convention but the query receives **no hidden edit** and has **no correct Q/K label**. That is much closer to a genuine no-signal test.

Even better would be a preregistered task with target / unrelated concept / random / none query conditions and an explicit `UNKNOWN`/`NEITHER` response or a frozen abstention threshold.

---

## 3. Twin-pair null correction

The twin construction itself is excellent:

- visible prompts are byte-identical;
- hidden interventions differ;
- required labels are opposite;
- both members must be correct for a twin-pair success.

But two different reference points have been conflated:

1. **Independent fair coin-flip benchmark:** probability both rows happen to be right = 0.25.
2. **Deterministic prompt-only/no-hidden-state strategy:** same visible prompt -> same greedy label on both twins, while the required labels are opposite -> twin-pair accuracy = **0.0**.

The experiments use deterministic greedy decoding, so 0.25 is not the structural prompt-only null. Binomial tests treating twin successes as independent Bernoulli(0.25) “chance” events are not justified by the design.

Recommended wording:

> “Twin-pair accuracy is 0.083, far below the same-exemplar anchor (0.521) and cheap reader (0.986), and close to the constant-label floor. A value of 0.25 is shown only as the independent-coin benchmark; the deterministic prompt-only null is 0.”

This correction does **not** damage the core held-out-semantic conclusion.

---

## 4. Handoff audit

### What the handoff does well

- records retractions rather than hiding them;
- keeps exact headline numbers and failure modes;
- distinguishes some discovery vs confirmation artifacts;
- explicitly narrows note 37 after the intended generalization fails;
- warns against several invalid comparisons already discovered in the project.

### Problems that matter now

#### A. Strategic contradiction

`HANDOFF.md` says:

- “No further LoRA training”; and
- the old adapters were never saved.

The current `scripts/run_remap_training.py` is an unrun v3 program that:

- trains new adapters;
- evaluates `none` and cross-layer transfer;
- explicitly calls `save_pretrained()`.

Either v3 is dead code under the standing constraint, or the constraint has changed. The handoff must make that decision explicit.

#### B. Stale v3 hypothesis text

The current v3 protocol builder still says a positive result would show that fixed-label training **degrades remapping**, even though the same source records that v1 falsified that exact hypothesis. This is a protocol-level contradiction and should be repaired before v3 can be considered frozen.

#### C. Random=noise semantics propagated into the handoff

Phrases such as “concept-versus-noise gap,” “destroys selectivity,” and the false-positive safety analogy should be replaced by the target-versus-arbitrary-direction codebook distinction above.

#### D. Semantic abstraction remains overstated

The handoff still contains language like “the model reads meaning, not just disturbance.” Note 23 is direct evidence that the strong earlier effect need not be semantic abstraction. Prefer:

> “The model discriminates previously demonstrated concept-derived axes better than matched random axes; held-out category generalization fails under the tested interface.”

#### E. Frontier is not explicit enough

After the note-29 -> note-37 descendant chain repeatedly narrows its predecessor, a Research-OS handoff should root-rebuild rather than name another small prompt variant. It should explicitly mark branches **PROMOTE / LIVE / PRUNE / REPAIR ONCE** and choose one highest-EVI next action.

---

## 5. Publication/audit state

`spar-application/AUDIT-MANIFEST.md` is obsolete for the current tree.

It is stamped 2026-08-10 and explicitly says anything added after the audit date is not audited. It records older test counts and a state before the subsequent note-29--37 work/current runner changes. It should not be presented as certifying the current repository.

Before mentor/application exposure, either:

- rerun/re-stamp the release audit on the current tree, or
- mark the old manifest as historical/obsolete.

---

## 6. Literature gaps that materially change framing

The formal `PAPERS-REVIEWED.md` does not contain dedicated entries for several directly relevant works, even though some are cited inside cached papers.

### Highest priority

1. **Li Ji-An et al. (2025), arXiv:2505.13763 — “Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations.”**
   - Very close prior to the codebook-ICL branch: models learn to report/control activation directions from sentence-label demonstrations; performance depends on semantic interpretability and explained variance.
   - This should be read before claiming novelty for demonstration-based arbitrary activation reporting.

2. **Binder et al. (2024), arXiv:2410.13787 — “Looking Inward: Language Models Can Learn About Themselves by Introspection.”**
   - Uses a self-vs-other comparator for privileged behavioral self-prediction; positive on simpler tasks, weak on harder/OOD tasks.
   - Directly relevant to defining the stronger target than “a hidden state is behaviorally usable.”

3. **Plunkett et al. (2025), arXiv:2505.17120 — “Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions, and Improve with Training.”**
   - Especially relevant to the proposed “does introspection training produce generalization?” experiment.

4. **Song, Lederman, Hu & Mahowald (2025), arXiv:2508.14802 — “Privileged Self-Access Matters for Introspection in AI.”**
   - Gives the cost-matched third-party comparator criterion that the reader experiments should use as the conceptual standard.

5. **Song, Hu & Mahowald (2025), arXiv:2503.07513 — “Language Models Fail to Introspect About Their Knowledge of Language.”**
   - Negative comparator evidence; important for avoiding equating good prompted judgments with privileged self-access.

### Useful second tier

- Nicolas Martorell (2026), arXiv:2603.18893 — quantitative/logit self-report tracks probe-defined internal states and can be causally steered.
- Naphade et al. (2026), arXiv:2603.20276 — Introspect-Bench / policy-access framing.
- Liu et al. (2026), arXiv:2607.11881 — current metacognition survey/citation map.

The literature ledger itself notes that its searches occurred after the experiments and that systematic citation chaining has not been completed. Treat novelty as provisional until this pass is done.

---

## 7. Research-OS frontier after this audit

### PROMOTE

**P1. Causal hidden-state codebook access exists under matched visible prompts.**  
Evidence: codebook ICL confirmation with exact twins and controls.

**P2. Training robustly improves reporting of weak target directions.**  
Evidence: multi-seed remap result at strengths 0.25/0.15.

**P3. Under the tested interface, held-out semantic category generalization fails despite near-perfect cheap-reader access to the relevant category geometry.**  
This is the strongest current result for the core research direction.

### LIVE / unresolved

**L1. Does introspection training produce semantic generalization rather than only exact-axis/arbitrary-codebook generalization?**

**L2. Does training harm true no-signal specificity / calibrated abstention?**  
Current random arm does not answer this.

**L3. Is there privileged self-access after cost matching?**  
Reader comparisons constrain this, but the project should use the explicit comparator criterion rather than treat “reader wins” as a universal philosophical conclusion.

### PRUNE

**R1. Fixed-label training damages remapping.**  
Refuted by v1; current v3 `claim_boundary` must stop reviving it.

**R2. Note-14 style same-axis performance demonstrates semantic abstraction.**  
Killed by held-out semantic experiment.

**R3. The prompt-conflict branch deserves more one-line descendants right now.**  
Note 37 narrows it to carrier wording; after the long chain, marginal EVI is low.

**R4. Current random-direction results establish false positives / “confidently wrong about concept X.”**  
Not tested by the task.

### REPAIR ONCE

**Q1. Public and handoff epistemic consistency.**  
Repair the stale claim language and audit stamp once, globally, before doing more experiments.

---

## 8. Highest-EVI next action

### First: no-compute epistemic repair

Before another GPU minute:

1. Correct the random-arm interpretation everywhere public-facing.
2. Correct the twin-pair null language.
3. Demote note-14 “semantic abstraction” wording in README/handoff/claims.
4. Resolve “no more LoRA” vs unrun v3 explicitly.
5. Remove/fix v3's stale fixed-label-degrades-remapping `claim_boundary` before any run.
6. Mark the Aug-10 audit manifest obsolete or re-stamp it on the current tree.
7. Add the high-priority missing papers to the formal literature ledger and write explicit novelty deltas.
8. Stop the note-38 prompt descendant branch.

### If one final LoRA run is allowed

Use it to kill the root uncertainty, not to continue the prompt branch:

**Primary:** train/save one reporter and rerun the frozen note-23 held-out semantic-generalization design after training.

Predeclare outcomes:

1. semantic improves while scrambled does not -> evidence training induces meaningful semantic generalization;
2. semantic and scrambled improve similarly -> geometric/arbitrary codebook generalization, not semantics;
3. both remain near floor while reader remains near 1 -> training teaches exact-axis/reporting behavior but not abstraction;
4. cheap reader still dominates after fair cost matching -> no evidence of privileged advantage under that comparator.

**Secondary, same checkpoint:** genuine no-signal specificity with target / unrelated / random / none queries and `UNKNOWN`/`NEITHER` or a frozen abstention rule.

**Third priority:** cross-layer transfer.

Save the adapter/checkpoint so every later measurement can be rescored without retraining.

### If the no-more-LoRA decision stands

Stop experimenting and write the SPAR application. A defensible concise story is:

> I built causal matched-visible tasks for hidden-state reporting and repeatedly attacked my own interpretations. The model can use injected activation directions, but cheap external readers often dominate it; a strong apparent semantic result collapsed under held-out exemplars even though the semantic category was almost perfectly readable. Training expands weak/arbitrary-axis reportability, while whether it creates genuine semantic generalization remains open.

That is a cleaner research story than “training causes false positives,” and it foregrounds the strongest evidence rather than the longest experiment chain.

---

## 9. Bottom line

The local agent was generally a **good experimentalist and a weaker research scheduler**. It caught several of its own confounds, preserved nulls, added increasingly strong controls, and the raw artifacts support most of the reported numbers. The failure mode was letting a useful descriptive effect acquire a stronger semantic/safety interpretation than the task measured, and then spending too many descendant experiments refining a narrowing prompt mechanism.

The repository does not need another broad exploratory pass. It needs:

1. one epistemic consistency repair;
2. one root-level decision about whether a final training experiment is worth reopening LoRA;
3. otherwise, application/write-up work.
