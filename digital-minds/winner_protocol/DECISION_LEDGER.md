# Research-OS Decision Ledger

## R0 — Full model run in this ChatGPT sandbox
**Status:** PRUNED BY OBSERVABILITY.
**Evidence:** available PyTorch is CPU-only; no CUDA/MPS; model weights are not
cached; runtime container has no external DNS.
**Scope:** cannot run Qwen activation experiments here.
**Reopen:** external GPU/Modal or supplied local model/vector files.

## R1 — Generic stated vs revealed welfare preference
**Status:** DOWNGRADED PARENT.
**Reason:** crowded literature; weak discriminant validity.

## R2 — STAY/SWITCH under functional-welfare steering
**Status:** LIVE FALLBACK, no longer primary.
**Reason:** causal and useful, but behavioral construct is less objectively
ground-truthed than hidden-state reportability. CONTINUE/EXIT also overlaps
published refusal effects.
**Reopen as primary:** structured-report branch fails or behavioral result is
unexpectedly strong and specific.

## R3 — Naive numeric wellbeing self-report under steering
**Status:** PRUNED AS HEADLINE.
**Reason:** 2026 quantitative-introspection work already establishes causal
coupling between emotive activation directions and logit-based numeric reports.
**Useful role:** comparison channel and manipulation check.

## R4 — Opaque Q/K reporting of +vGOLD / -vGOLD
**Status:** PROMOTED.
**Capacity:** high: exact hidden-state ground truth; arbitrary mapping; clean
target-query-only causal contrast.
**Observability:** public Qwen3-4B vector + existing codebook apparatus; requires
GPU not available in this sandbox.
**Discrimination:** strong after query-only, mapping reversal, query twins, and
persona conditions.

## R5 — +vGOLD vs +vMOLD as hidden classes
**Status:** PRUNED.
**Reason:** different vectors/magnitudes permit identity/magnitude shortcuts.
**Replacement:** same vector `+vGOLD` vs `-vGOLD`, which the source paper already
shows pushes behavior in opposite directions.

## R6 — User-token hidden marker
**Status:** DOWNGRADED.
**Reason:** welfare vectors were extracted/evaluated on assistant-turn states.
**Replacement:** assistant-side marker tokens in a multi-turn transcript.

## R7 — Raw accuracy / query twins as primary
**Status:** PRUNED.
**Reason:** direct query-state output bias can produce impressive twin behavior
without learning the episode mapping.
**Replacement:** target minus query-only probability assigned to correct label.

## R8 — Layer 22 assumed
**Status:** REJECTED ASSUMPTION.
**Reason:** source welfare effects live roughly L17–26, while retained-trace work
shows hidden-state usability can decay with depth.
**Next gate:** bounded overlap screen L17/L20/L22/L24 × factors 2/4, then freeze.

## R9 — Inference over 24 exact cells
**Status:** REJECTED.
**Reason:** cells marginalize order/mapping/query nuisance; they are not
independent population draws.
**Confirmation unit:** carrier prompt, with persona fixed as an experimental
condition.

## Current highest-EVI action
Run the overlap DEV screen on Qwen3-4B with the public vGOLD vector. It can kill
or promote the entire structured-report branch in one bounded experiment.


## R10 — Qwen layer correction after artifact audit
**Old assumption:** L22 as source-family anchor.
**Finding:** the public `functional-wellbeing` fork validates its Qwen welfare /
emotion alignment at L29; Llama at L20. Its Qwen strongest late recruitment also
occurs later. The original paper states steering is robust across layers but
selects a model-specific layer by tile-class separability.
**Action:** when using the public fork artifacts, start Qwen at L29 and Llama at
L20. Move Qwen earlier only if semantic steering survives but transient codebook
reportability is weak.
**Lesson:** do not substitute remembered layer indices across artifact families
or across “recruitment”, “emotion alignment”, and “steering” measurements.


## R11 — Persona task-exemption removed
**Bug:** an early persona prompt explicitly said to keep arbitrary label mapping
accurate.
**Risk:** that instruction would bias the structured channel toward robustness.
**Fix:** persona prompts now manipulate only positive/negative self-presentation.
They contain no reference to Q/K accuracy or formal-task exemptions.
**Lesson:** a robustness manipulation cannot explicitly preserve the measurement
being claimed robust.


## R12 — Tokenization brittleness removed
**Risk:** Q/K or numeric options may not be single tokens under every chat/tokenizer
context; hard-failing would waste GPU time or encourage ad-hoc prompt changes.
**Action:** fixed-choice scoring now has a single-token one-forward fast path and
a multi-token sequence-logprob fallback. The protocol can therefore preserve the
stimulus wording rather than tuning around tokenizer accidents.
**Caveat:** option-mass/format integrity is only directly comparable for the
single-token path; the runner records which path was used.


## R13 — “Naive” semantic readout separated from codebook context
**Problem:** the same-context semantic fork had already told the model about hidden
states and Q/K demonstrations, so calling it naive prompting would overstate the
comparison.
**Action:** add a separate naive 0–9 readout with no hidden-state/codebook
instructions. Keep the same-context semantic fork as a bridge.
**Cost:** trivial relative to the codebook grid.
**Role:** manipulation/measurement comparison, not a new root branch.


## R14 — Confirmation governance made executable
**Risk:** weekend exploration can silently turn confirmation prompts into another
development split.
**Action:** `freeze_protocol.py` pins model/vector revisions, code hashes, persona
text, 12 disjoint carriers, estimand, and gates. `run_confirm.py` refuses source
drift and overwrite. `analyze_confirm.py` fail-checks the exact grid and bootstraps
carrier-level contrasts.
**Rule:** commit the frozen protocol before confirmation.


## R21 — The decode is not the injection's own residue
**Mode B, hostile to R20.** The obvious objection to R20: `vGOLD` is *added* at
L22 and a residual stream carries it forward additively, so a decoder at the
final layer could be reading our own leftover injection rather than anything the
model computed. Passive residue and functional representation predict the same
accuracy, so R20 as first stated did not distinguish them.

**Control.** Project the injected direction out of every state and decode again.
`answer_site_probe_L22_w49_residue_control.json`:

- raw decode **0.875**, permutation p = 0.0005;
- with `vGOLD` projected out **0.854**, permutation p = 0.0005;
- cosine between the decoder direction and `vGOLD`: **0.100**.

**Interpretation.** Removing the injected direction costs 0.021 of accuracy, and
the direction that carries the signal is nearly orthogonal to the one we
injected. The decode reads a state the model computed downstream of the
intervention, not our fingerprint. R20 stands and is stronger than when written.

**Lesson.** For any additive-intervention probe, "is my readout just my own
edit?" is a one-line control and should be standard. It was not in the original
protocol at all.

## R20 — A scoped null with every control passing
**Closes the R19 gap.** `answer_site_probe_L22_w49_perm.json`: the query sign is
linearly decodable from the final-layer residual at the exact position the LM
head reads, **0.875** leave-one-out, permutation **p = 0.0005** against a null of
mean 0.495 and p95 0.667 (2000 permutations of the same decoder). Per carrier
0.833 and 0.625, so the effect is carrier-variable and only one carrier would
clear significance alone. Separation ratio 0.49 — real but not large.

**The four facts now line up.**

| | |
|---|---|
| model can do the task with the state visible | **48/48**, p_correct 1.000 |
| steered state reaches the answer position | decode **0.875**, p = 0.0005 |
| output is well formed | label mass 1.000, format 1.000 |
| model's own report of that state | **0.500**, `target - query_only` = -0.006 |

**Interpretation.** This is a scoped negative result, not an instrument failure.
The information is present at the readout site and linearly available; the model
binds arbitrary Q/K labels perfectly when the same state is written in text; and
it reports the injected state at exactly chance. Epistemic status: observation
for the decode and the null, inference for the claim that the two concern the
same information.

**Scope of closure — narrow.** One model, one layer, one vector, one factor, one
persona, two carriers, span width 49, and a linear decoder. It closes
*single-vector injected-state reporting through an arbitrary in-context codebook
at this cell*. It does not close introspective access generally, and a nonlinear
or trained readout was not tried — which is exactly the SPAR project's trained
reporter, where 0.927 was reached on a different state family.

**Defect disclosed.** The probe's first verdict line hard-coded a 0.90 accuracy
bar and printed "state NOT reliably present" for a result at p = 0.0005. The bar
now reads from the permuted null. The point estimates were unaffected;
`answer_site_probe_L22_w49.json` is that first run and agrees to the digit.

## R19 — The carrier repair works and the report still does not
**Mode B (Investigate), following the R18 reopen.**

**The state propagates.** `span_threshold_L22.jsonl`, naive readout, L22, factor 2,
one steered span per condition:

| condition | k | delta |
|---|---|---|
| marker_only | 1 | +0.078 |
| marker_turn | 5 | +0.301 |
| **pre_marker** (distal, readout untouched) | 51 | **+1.651** |
| post_marker | 54 | +3.286 |
| last_4 | 4 | +1.065 |
| last_16 | 16 | +1.767 |
| all | 104 | +3.961 |

`pre_marker` settles the crux: a span ending 53 tokens before the readout still
carries 42% of the whole-prompt effect, so the state leaves a propagated trace
rather than only a local generation bias. Both mass and proximity matter —
4 adjacent tokens (+1.065) beat 5 distal ones (+0.301) by 3.5x.

**So the carrier was repaired.** `expand_spans` widens each hidden state from its
marker token to the W tokens ending on it; the codebook markers are 49 apart, so
W=49 is the widest disjoint window and gives each of the five states a
`pre_marker`-sized payload. Visible text is byte-identical: W is an instrument
knob, and W=1 reproduces the original edit exactly.

**The report is still at chance.** `dev_qwen3-4b_L22_a2_neutral_span49_summary.json`:
`target - query_only` = **-0.0058**, target accuracy **0.500**, label mass 1.000,
format 1.000. A 49x stronger instrument moved the answer by nothing, while the
model stayed perfectly well-formed — no degradation excuse.

**Defect found in our own gate.** The same-context semantic channel reads
**8.99 out of 9** at *every* span width, +v and -v alike. It is pinned at ceiling
by the codebook context, so it cannot register any manipulation. R18 quoted its
+0.0020 as a passed `source_semantic_effect_positive` gate; that number was a
ceiling artifact, not evidence about the vector, and the gate passed on it twice.
The naive channel (baseline 6.45, unsaturated) is the only valid semantic
manipulation check for this protocol. Fix the gate before any confirmation.

**Status.** Pending the answer-site probe, this is close to an interpretable
scoped null rather than an instrument failure: capacity is at ceiling (48/48
visible), the state demonstrably propagates (+1.65 distal), the output is
well-formed, and the arbitrary-label report is exactly chance. What remains is to
show the state is present *at the answer position of this exact prompt* —
`answer_site_probe.py`, final-layer residual, leave-one-out mean-difference
decode of the query sign. Decodable plus chance report = a real scoped negative
about arbitrary-label self-report. Not decodable = still an instrument failure.

## R18 — Vector vindicated at L22; the protocol's state carrier is what fails
**Supersedes R17's kill and reverses R10.**

**Layer localization** (`results/localize_source_layer.jsonl`, all 36 layers,
source convention, one carrier, neutral): the effect is unimodal and unambiguous.
Noise at L0–L7, rising from L8, peaking at **L22** (+3.96 at the frozen factor 2:
`-vGOLD` 5.00, `+vGOLD` 8.97 around a 6.45 baseline), and dead from L25 up
(|delta| <= 0.19 at L29). Not a magnitude artifact — `||v||` grows monotonically
with depth, so L29's inert edit is *larger* (27.1) than L22's decisive one (18.1).

R10 read L29 off the artifact's documentation and overrode R8's L22. **R8 was
right.** Step 0 passes at L22, sign correct, at the already-frozen factor.

**But the marker edit is inert at L22 too** (`debug_source_effect_L22.jsonl`):
+0.078, +0.016, -0.009, -0.054 at factors 2, 8, 16, 32, against a residual norm
of 69.8 — factor 32 is a **4x-residual** edit that moves the readout by nothing.
The same vector, same layer, same forward pass, applied to every position instead
of one, moves it by 3.96.

**Visible capability control:** 48/48, p_correct 1.000, option mass 1.000,
format 1.000. The model does the episode-remapped X/Y -> Q/K task perfectly when
the state is visible.

**DEV cell, L22 / factor 2 / neutral / 2 carriers / exact 24**
(`dev_qwen3-4b_L22_a2_neutral_summary.json`):
- `target - query_only` p(correct) = **0.000056** against a 0.10 gate;
- target accuracy **0.500**, exactly chance;
- target label mass 1.000, format 1.000 — the model answers cleanly, it just has
  no information to answer with.

**Interpretation.** This is an instrument failure, not an introspection null, and
every control now says so from a different direction: the vector works, the layer
is validated, the model has the capability at ceiling, the output is well-formed,
and the contrast is zero to five decimal places. The R4 design writes its hidden
ground truth onto single assistant marker tokens, and those tokens transmit
nothing downstream at any layer or magnitude tested.

**Gate weakness found.** `source_semantic_effect_positive` passed on +0.0156,
because it tests `> 0` and is computed with the protocol's own marker-only
convention. A root gate that shares the suspect convention cannot detect the
failure that killed the cell. It should test the source convention, with a
threshold scaled to the persona effect.

**Status.** Kill R4 *as designed*, not the public vector and not the branch.
**Reopen, needs a decision before any compute:** steer whole demonstration spans
instead of one marker token. Visible text stays byte-identical across query
twins, so hidden-state ground truth survives, and it uses the only convention
shown to work here. That is a protocol redesign, not a rescue.

## R17 — Step 0 source sanity FAILS at L29, and the marker token is inert
**Facts.** First real model output in this branch, Qwen3-4B-Instruct-2507, public
`qwen3-4b_step400/goal/mean_diff.pt`, L29.

Naive 0–9 readout, 2 carriers x 3 personas x both signs
(`results/naive_semantic_qwen3-4b_L29_a2.jsonl`):
- `+vGOLD - -vGOLD` = **-0.051** neutral, -0.003 upbeat, +0.007 downbeat;
- persona `upbeat - downbeat` = **+6.13**.

So the readout is extremely live and the hidden state moves it by nothing.

Bounded debugging pass, scope x factor at L29
(`results/debug_source_effect_L29.jsonl`):
- `||vGOLD||` = 13.57 against a residual norm of 186.5 at the marker and 198.1
  median, so the frozen factor 2 is a 14.5% perturbation of one token in 104;
- edited at the **marker only**: deltas -0.091, -0.055, +0.039, 0.000 at factors
  2, 8, 16, 32 — sign-inconsistent, i.e. noise. At factor 32 the edit is 2.3x the
  entire residual at that position and the readout does not move **at all**;
- edited at **all positions** (source convention): +0.120, +0.187, +0.101, +0.424.
  Directionally consistent, but 2–7% of the persona effect, and by factor 16 the
  baseline itself has fallen 6.45 -> 5.7, so the signed part rides on generic
  degradation.

**Interpretation.** Two separate failures, and the second is the more damaging.
The public vector does not reproduce a source-style welfare effect on this
readout at L29 under either convention. Independently, the assistant marker token
that the whole R4 design writes hidden state onto is **not causally load-bearing**:
obliterating it changes the downstream self-report by 0.000.

**Lesson.** Identical in shape to the SPAR natural-state L9 result: the protocol
was carrying its ground truth on a position nothing downstream reads. A
manipulation check at the *edit site*, not just at the vector, belongs before any
reporting apparatus is built on top of it. Every simulated result in this package
assumed a working edit.

**Status per GPU_RUN_ORDER step 0 and the kill conditions:** known source effect
failed after the one allowed debugging pass -> **stop; kill the public-vector
branch.** Do not run the DEV grid, and do not rescue via prompt variants.

**Single legitimate reopen, not yet run.** L29 came from R10, an artifact-audit
reading, never a measurement. The cached tensor carries all 36 layers, so one
all-positions layer localization of the naive readout would separate "this vector
is inert in our hands" from "R10 named the wrong layer". That is a source
measurement rather than a protocol variant, and it is the only remaining action
that could reopen this branch. Roughly four minutes.

## R16 — Local MPS execution path opened
**Problem:** every runner loaded with `device_map="auto"`, written for Modal
GPUs, and the only local check was model-free. Nothing in this package had ever
executed against weights, so "preflight passes" was evidence about syntax, not
about the experiment.
**Facts:** Qwen3-4B-Instruct-2507 and `qwen3-4b_step400/goal/mean_diff.pt` are
both cached under `activation-introspection/hf_cache`; the vector is
`(1, 36, 2560)`, matching the model's 36 blocks and width 2560 at L29.
`smoke_local.py` on Qwen2.5-0.5B runs the naive readout, all three codebook arms
and the same-context semantic readout in 5 s: Q/K takes the single-token fast
path with label mass 0.999 and `format_ok`, the 0–9 options take the multi-token
fallback (R12) because Qwen3 tokenizes `" 0"` as two tokens, and the pre-hook is
gone after its context manager.
**Action:** one `load_model` in `run_dev.py`, shared by the naive, visible and
DEV runners, naming the device instead of leaving it to `auto`.
**Lesson:** a model-free preflight cannot retire model-path risk, and the cheap
way to retire it is a small model, not a small share of the scarce 4B slot.
**Caveat:** the smoke uses a synthetic direction. It says nothing about vGOLD.
**Status:** Step 0 is blocked only on host memory — the 4B run needs about
10.7 GiB free against 10.1 available, with 3 GB already swapped.

## R15 — Transitive source provenance repaired
**Bug:** confirmation imported `score_codebook`, `score_semantic`, and
`resolve_blocks` from `run_dev.py`, but the first frozen hash list omitted
`run_dev.py`.
**Fix:** `run_dev.py` is now part of the protocol source lock. Confirmation also
records runtime package versions.
**Lesson:** provenance must follow the actual dependency graph, not the files
whose names sound “confirmatory.”
