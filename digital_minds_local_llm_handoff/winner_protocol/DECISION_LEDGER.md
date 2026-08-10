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


## R15 — Transitive source provenance repaired
**Bug:** confirmation imported `score_codebook`, `score_semantic`, and
`resolve_blocks` from `run_dev.py`, but the first frozen hash list omitted
`run_dev.py`.
**Fix:** `run_dev.py` is now part of the protocol source lock. Confirmation also
records runtime package versions.
**Lesson:** provenance must follow the actual dependency graph, not the files
whose names sound “confirmatory.”
