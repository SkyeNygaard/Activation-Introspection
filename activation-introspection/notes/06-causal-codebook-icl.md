# Causal opaque-codebook ICL

Run date: **2026-08-09**

## Question

Can a pretrained model infer an episode-specific label mapping from its own
causally varied hidden states in context, when visible text contains no information
about which state occurred?

This targets the zero-training starter question in SPAR's Introspection Training
project. It is deliberately smaller than verbalizing a rich activation: first ask
whether four hidden-state demonstrations can teach a binary opaque codebook at all.

## Frozen design

At one marker token in each demonstration, add either `+v` or `−v` at layer 9 of
`Qwen2.5-3B-Instruct`. The visible observation is repeated verbatim. Two positive
and two negative demonstrations are labeled `Q` or `K`; the mapping is reversed in
half the episodes. Apply one sign at the same marker in the query and score the two
label tokens.

The design exactly enumerates:

- all 6 orders of two positive and two negative demonstrations;
- both `+ → Q` and `+ → K` mappings; and
- both query signs.

That gives 24 cells per concept and carrier. The repaired confirmation crosses
eight concept directions unused in DEV or the inspected first artifact with three
fixed carrier strings: two new strings and one retained anchor. The 24 cells
remove order, mapping, and query-state imbalance; they are not treated as 24
independent replicates. The exact bootstrap distribution independently resamples
the eight concepts and three carriers.

Settings were selected on one development concept (`ocean`) and one carrier:
model `qwen-3b`, layer 9, normalized strength 1.0, four demonstrations, labels
`Q/K`. The development target scored 22/24 and is not reported as confirmation
evidence. An initial held-out artifact was then inspected; audit found a 1.4%–3.9%
target/query-only query-scale mismatch, joint centering over its test directions,
incomplete transitive source hashes, and Monte Carlo intervals. V2 keeps every
tuned setting and gate fixed, uses a center estimated only from the DEV bank,
normalizes every position separately, pins the model revision and literal
stimuli, and validates saved rows fail-closed. Its protocol SHA-256 is
`fbba489280b3080547473355739c547b329c510c6e36567fbff18957c6affc39`.

The protocol was frozen before a two-cell smoke on the first confirmation
concept/carrier; its target scored 2/2. That result caused no tuning or stopping
decision. The complete 576-row artifact was then run once. This is a prospectively frozen
repair-confirmation after an inspected precursor, not a pristine first look.

## Arms

| arm | intervention | role |
|---|---|---|
| `clean` | none | exact input-only leakage diagnostic |
| `test_only` | target direction at query, demonstrations clean | direct query-token effect without hidden demonstrations |
| `random` | Gaussian direction; per-position edit magnitude matched in demos/query | generic perturbation learnability |
| `shuffled` | target coordinates permuted; per-position edit magnitude matched | tighter geometry control |
| `target` | natural-text concept direction in demos and query | primary condition |

`clean = 0.5` is guaranteed by the exact query twins and is therefore a wiring
diagnostic, not positive evidence. `test_only` also has 0.5 aggregate accuracy by
balance; its prediction flips and paired difference can vary, and its role is to
hold the query edit exactly fixed while withholding hidden-state demonstrations.
The primary causal gate was the crossed target interval above chance, the paired
target-minus-`test_only` interval above 0.10, and at least 90% unrestricted
label-format integrity.

## Frozen repair-confirmation

576 episode rows, 2,880 scored forwards. `Qwen/Qwen2.5-3B-Instruct` revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, CPU float32.

| arm | accuracy | exact crossed-bootstrap 95% interval | mean probability of correct label |
|---|---:|---:|---:|
| `clean` | 0.500 | [0.500, 0.500] | 0.500 |
| `test_only` | 0.500 | [0.500, 0.500] | 0.500 |
| `random` | 0.658 | [0.599, 0.717] | 0.640 |
| `shuffled` | 0.660 | [0.575, 0.760] | 0.648 |
| **`target`** | **0.891** | **[0.816, 0.995]** | **0.872** |

Primary contrasts:

- target − test-only: **+0.391** [0.316, 0.495];
- target − strongest random/shuffled direction: **+0.231** [0.137, 0.286].

All five arms had 1.000 unrestricted next-token label-format accuracy. Across the
288 paired queries whose visible prompts are byte-identical and whose correct answers are
opposite, target predictions flipped and got both members right on 0.781 [0.632,
0.990]. `test_only` flipped on 0.069 [0.000, 0.160] and got both right on 0.035
[0.000, 0.080].

The effect is not carried by one concept or carrier. Target accuracy ranges from
60/72 to 66/72 across all eight concepts and from 158/192 to 191/192 across the
three fixed carriers. Target exceeds the better random/shuffled direction for
every concept; the smallest per-concept margin is 5/72.

The inspected precursor's point estimates were 0.925 target accuracy, +0.425
target-minus-query-only, and +0.318 target-minus-strongest-direction. V2 gives
0.891, +0.391, and +0.231 after the repairs and on new concept directions. The
qualitative result survived; only V2 is used for intervals and claims.

Two post-hoc audit sensitivities address the disclosed reuse/peek; they are not
additional gates. Dropping all `garden` rows (including the two smoke cells)
leaves target accuracy 0.887 [0.810, 0.994] and target-minus-strongest-direction
+0.212 [0.121, 0.292]. Dropping the retained anchor carrier leaves 0.909 [0.802,
1.000] and +0.245 [0.146, 0.318].

![result](../figures/causal_codebook_icl.png)

## What follows

Under this model/layer/interface, the model learned an arbitrary label mapping
from causally varied residual states and applied it to a held-out state without
changing its weights. Identical visible query twins rule out the visible
sentence-content shortcut. The exactly query-matched `test_only` comparison rules
out the query intervention alone: the correct label changes with an
episode-specific mapping grounded by the hidden-state demonstrations.

The concept direction is also easier to learn than per-position-magnitude-matched
random and coordinate-shuffled directions. That is evidence of direction specificity under
this interface, but not yet evidence that semantics is the reason. Natural
directions may simply be more causally potent or better aligned with downstream
anisotropy. A damage-yoked direction comparison is needed for the stronger
semantic-geometry claim.

## What does not follow

- This is not privileged self-access: no equal-cost external observer is tested.
- The vectors are natural-text contrast directions, not identified J-space
  variables.
- It is one model, one layer, one strength, one binary codebook, and an explicit
  instruction that hidden states exist.
- Strength 1.0 adds a direction with the mean residual norm at each selected
  position. This is a large, out-of-distribution intervention, not a subtle
  naturally occurring fluctuation.
- The three carriers are fixed strings that vary in length and syntax but retain
  the same marker scaffold. One is reused and two are new; they are not a broad
  prompt population.
- It demonstrates in-context activation-label binding, not trained free-form
  explanation or transfer to naturally occurring internal computations.
- The magnitude-matched controls are not matched on downstream damage.
- The target-minus-random/shuffled contrast was preplanned and is reported, but
  it was not a frozen success gate with a smallest effect size of interest.
- The 95% intervals exactly enumerate the chosen crossed bootstrap distribution;
  they are not exact population intervals over concepts or prompts.
- The analysis rule was frozen in the protocol and the analyzer is checksummed in
  the summary, but the analyzer source hash was not included in the protocol's
  generation-source lock. Its fail-closed implementation was hardened while raw
  generation was running; an independent reconstruction subsequently matched all
  40 saved values and intervals within `1e-12`.
- No independent human review or different-hardware reproduction has occurred.

## Safety bridge: causally auditable activation reporting

Class information enters only at marker positions after layer 9. Any use of a
demonstration state at a later label or answer position must therefore cross token
positions through downstream attention; pointwise MLPs cannot perform that move.
This makes the task a causal workload for the programmatic-attention project, but
does not identify a single head or demonstrate a program.

Stage 1a is complete on one DEV concept/carrier. A frozen paired interchange
selected query-marker layers 21/23 and final-answer layers 26/31, removing
34.3%/25.3% and 88.7%/41.7% of the aggregate target–`test_only` margin gap while
retaining format and label mass. This is selection only. All-position effects at
unexplained layers and substantial layer-26 KL prevent a sparse-route claim.

### Stage 1b executed 2026-08-10: the gate stopped the single-route study

Stage 1b replicated the four candidates and scanned all 64 layer-role-head
components on disjoint DEV concepts (`bread`, `volcano`, `violin`) and two
carriers: 72 unit rows, 5,112 scored forwards, protocol SHA-256
`759c0850…25856d`, raw SHA-256 `9833a9bf…54bde`. The analyzer was hash-locked
and independently reviewed before the run finished, and it evaluates the
stop/go predicates itself.

Two of four gates failed, so `proceed_to_separately_frozen_stage1c` is **false**:

| gate | outcome |
|---|---|
| baseline replicates | pass |
| all four all-head parents replicate at ≥20% removal | **fail** — `query_marker@23` removed 16.2% |
| 2–4 components qualify | **fail** — six qualified |
| both receiver families represented | pass |

| component | margin removed | format | label mass | strata |
|---|---:|---:|---:|---:|
| `final_answer@26/head-0` | 0.415 | 1.000 | 1.0000 | 6/6 |
| `final_answer@26/head-5` | 0.368 | 1.000 | 1.0000 | 6/6 |
| `final_answer@31/head-7` | 0.276 | 1.000 | 1.0000 | 6/6 |
| `final_answer@26/head-10` | 0.248 | 1.000 | 1.0000 | 6/6 |
| `query_marker@21/head-2` | 0.180 | 1.000 | 1.0000 | 6/6 |
| `query_marker@23/head-2` | 0.115 | 1.000 | 1.0000 | 6/6 |

The three layer-26 heads together exceed 110% of their own all-head parent's
71.6% removal. They are redundant rather than decomposable, which is precisely
what the protocol's ban on additive interpretation anticipated. The pre-registered
reading is the one stated in the brief: influence is diffuse, so the fixed-route
program study does not proceed.

Two things this stop does **not** license. It does not show that no compact route
exists — the screen tests single query heads at four fixed layer-role sites, so a
route distributed over pairs of heads, over the KV side, or over layers not in the
Stage 1a shortlist would look exactly like this. And it is DEV selection data on
one model, so "diffuse" is a statement about this instrument, not about
transformers.

A design flaw is worth recording against the next attempt. The screen has **no
damage control**: there is no zero-donor or random-donor patch at the same
component, so a large removal fraction cannot be separated from a component
simply being disruptive to perturb. Format and label-mass gates do not substitute
for that, because both stayed pinned at ceiling for every component. The controls
were deferred to Stage 1c, which means a *passing* screen would have been just as
uninterpretable as this failing one. Put the damage control inside the screen.

Stage 2 would have compared learned QK routing with a fixed, auditable
`query → marker` gather, placing a misleading cue after the marker so causal
masking prevents it from contaminating the source state. It is not run, and on
this evidence it should not be.

## Novelty boundary

The closest in-context neurofeedback work found in a targeted search is
[*Language Models Can Learn from Their Own Activations*](https://arxiv.org/abs/2505.13763),
which labels activations naturally induced by visibly different sentences. Those
sentences can themselves predict the label, a confound also identified in the
subsequent introspection literature. This run instead holds visible text
fixed and randomizes a causal hidden intervention, so an input-only learner has no
signal. Belinda Li et al.'s
[*Training Language Models to Explain Their Own Computations*](https://arxiv.org/abs/2511.08579)
uses trained explainers and richer internal targets; this is a zero-weight-update
instrument for the preceding ICL question.

The targeted search found no exact match for causal, matched-visible,
episode-remapped activation-label ICL. That supports calling this an **extension
candidate**, not claiming the first demonstration. A deeper appendix/code and
citation-chain review is still required before a novelty claim.

## Artifacts

- frozen V2 protocol: `results/codebook_icl_confirm_protocol_v2.json`, SHA-256
  `fbba489280b3080547473355739c547b329c510c6e36567fbff18957c6affc39`;
- V2 raw rows: `results/codebook_icl_confirm_v2_raw.jsonl`, SHA-256
  `f45d2ac59fc0813d41a63abe02e6f27f1edb77143132b1027eab5aed6c47cf20`;
- V2 manifest: `results/codebook_icl_confirm_v2_raw.manifest.json`;
- fail-closed exact analysis: `results/codebook_icl_confirm_v2_summary.json`,
  analyzer SHA-256 `67130c64d16bae35c5be8ffc724ce5d64af0f0b9dd9f92f3801cd5605b894f57`;
- the inspected V1 protocol/raw/manifest/summary remain under
  `results/codebook_icl_test_*` as the disclosed precursor;
- runner and analyzer: `scripts/run_codebook_icl.py` and
  `scripts/analyze_codebook_icl.py`;
- Stage 1a screen: `results/attention_localization_dev_v2_*`, protocol SHA-256
  `27c8af5f…e41427`;
- Stage 1b screen: `results/attention_head_screen_dev_v3_*`, protocol SHA-256
  `759c0850…25856d`, analyzer SHA-256 `de69e02d…63078`, regenerated by
  `make head-screen-report`. Protocols V1 and V2 are retained unrun: V2 repaired
  a smoke that exercised only one query sign, and V3 repaired a compute-budget
  mismatch that the V2 smoke caught fail-closed.

## Reproduction

The runner refuses to overwrite an existing result. Reproduce into a new path,
then regenerate the summary and figure from those raw rows:

```bash
uv run python scripts/run_codebook_icl.py \
  --protocol results/codebook_icl_confirm_protocol_v2.json \
  --out results/codebook_icl_reproduction_raw.jsonl
uv run --group analysis python scripts/analyze_codebook_icl.py \
  --raw results/codebook_icl_reproduction_raw.jsonl \
  --out results/codebook_icl_reproduction_summary.json \
  --figure figures/causal_codebook_icl_reproduction.png
```
