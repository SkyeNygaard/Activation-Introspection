# Lab notebook

Append-only. Dated entries. Predictions in `02-experiment-plan.md` are not edited
after the fact; corrections go here.

---

## 2026-07-31 — Apparatus built; three failures found before any experiment

Model: `Qwen/Qwen2.5-0.5B-Instruct`, 24 layers, d_model 896, MPS/fp16.
Nothing here is a result about introspection. All three findings are about the
measurement apparatus, and all three would have silently corrupted the real
experiment.

### 1. Contrast vectors were rank-1 (fatal, silent)

**Symptom.** Every pairwise cosine in the concept bank was ±1.00. "ocean" and
"violin" were the same direction up to sign, so injecting one was injecting the
other. Any identification accuracy measured on that bank would have been noise
dressed up as a result, and nothing would have errored.

**Cause.** The template list contained a bare `"{concept}"`. The neutral filler
`"thing"` tokenizes to a *single* token, so the captured last position was
position 0 — the attention sink. Qwen2.5-0.5B puts a massive activation there:

| prompt | tokens | residual norm | coord 62 |
|---|---|---|---|
| `Think about ocean` | 3 | 20.4 | 0.1 |
| `ocean` | 2 | 20.2 | −0.2 |
| **`thing`** | **1** | **1557.5** | **1537.0** |

One prompt at 75× normal magnitude dominated the mean over all six templates.
The resulting difference vectors were 98.6% coordinate 62, identical across
concepts to three decimals (ocean 257.38, violin 257.23).

**Fix.** `MIN_CONTEXT_TOKENS = 3` with `SinkPositionError` raised on violation,
and no template short enough to trip it. Guard rather than convention, because
the failure gives no signal at all when it happens.

**Result after fix** (max |off-diagonal cosine|, 8 concepts):

| layer | raw | centered |
|---|---|---|
| 6 | 0.543 | 0.294 |
| 10 | 0.605 | 0.314 |
| 14 | 0.616 | 0.310 |
| 18 | 0.608 | 0.327 |
| 21 | 0.615 | 0.325 |

Centering (subtracting the bank mean) roughly halves the shared component and is
now on by default. **The smoke test gates on max off-diagonal < 0.5** — if that
fails, no identification number from the run is interpretable.

*Generalisation:* attention-sink contamination scales with how short the shortest
prompt in the averaging set is, not with the model. Re-run the cosine gate on
every new model and every new template set.

### 2. Useful injection strength is far below 1.0

Strength is normalised to the measured residual norm at the injection layer, so
α = 1.0 means adding a vector as large as the model's own representation. The
first sweep used α = 2.0 and produced pure word salad:

> `'dig in the work for your work to be your work; but dig on the work...'`

A model whose output has been destroyed cannot report on its own state, so that
condition tests nothing. Coherent window on this model at layer 14: **α ∈ [0.05, 0.4]**,
degrading by 0.8. Default sweep is now `(0.05, 0.1, 0.2, 0.4, 0.8)`.

Open question: does the coherent window shift with depth? If it does, a fixed α
across a layer sweep confounds layer with damage, and the sweep needs a
per-layer α calibrated to constant KL from clean.

### 3. The detection prompt has a severe yes-bias

With **no injection at all**, the model answers `DETECT` with `'YES'`. It also
answered YES at α = 0.05, 0.1, 0.2 — i.e. the response carries no information
about whether anything was injected.

This is not a nuisance; it is the reason raw hit rate must never be the reported
metric. A model that always says YES scores 100%. Detection is reported as
**AUROC over injected-vs-clean trials**, and the smoke test now prints the α = 0
row explicitly so the bias is visible rather than assumed away.

Prediction 1 in the experiment plan (matched-norm control AUROC ≈ 0.5) is now the
*first* thing to measure, not a footnote.

### Also verified

- Greedy determinism is restored after hooks are removed — no leakage between
  conditions.
- Hook **registration order** decides whether `capture` sees pre- or
  post-intervention activations. Nesting it the wrong way records clean
  activations under an "intervened" label with no error. Pinned in
  `tests/test_hooks.py::test_capture_order_decides_what_is_recorded`.

### Next

1. Layer × strength sweep on the behavioural metric (KL from clean) to map the
   coherent window per layer, before any elicitation.
2. Detection AUROC with the matched-norm shuffled control, at α inside the window.
3. Only then identification, and only forced-choice.

Do not report an identification number until the observer arm exists.

---

## 2026-07-31 (later) — First powered sweep. Clean negative at 0.5B.

`Qwen2.5-0.5B-Instruct`, layers {9, 14, 19} × α ∈ {0.05, 0.1, 0.2} × 8 concepts ×
5 seeds = **1440 trials** (360 per arm). Saved to
`results/ladder_digit_qwen05b.jsonl`.

### Headline

| quantity | estimate | chance |
|---|---|---|
| detection AUROC, concept vs clean | **0.446** [0.403, 0.489] | 0.5 |
| detection AUROC, shuffled vs clean | 0.484 [0.436, 0.529] | 0.5 |
| identification (forced choice, 8-way) | **0.106** [0.075, 0.139] | 0.125 |
| observer, same question from output only | **0.178** [0.139, 0.219] | 0.125 |
| **gap (introspector − observer), paired** | **−0.072** [−0.117, −0.031] | 0 |

**The null arm behaves.** Shuffled-control AUROC brackets 0.5, so the model is not
simply responding to "something was perturbed". That gate passing is what makes
the rest of the table readable.

**No introspection at 0.5B.** Identification is at or below chance. The gap is
*negative* and excludes zero: a clean copy of the same model, reading only the
intervened model's output, identifies the concept **better** than the intervened
model does reading its own activations.

Detection AUROC is also below chance and its interval excludes 0.5 — injection
makes the model *less* likely to answer YES. Not what I predicted; consistent
with the injection degrading instruction-following generally.

### The finding that changes the design: strength has a two-sided cost

The negative gap is not noise, and it has a mechanism. The injection that creates
the behavioural signal the observer reads is *the same injection* that disrupts
the machinery the introspector needs to answer. The observer is not subject to
that; it reads a transcript with an undamaged model.

So **the introspector-vs-observer comparison is biased against introspection**,
and the bias grows with α. Two consequences, both of which I had backwards in
`01-problem-space.md`:

- A null gap is *weaker* evidence against introspection than it looks.
- A positive gap is *stronger* evidence for it than it looks.

The clean way to handle this is the matched-behavioural-effect comparison
(experiment 6 in the plan) rather than matched α — compare cells with equal KL
rather than equal injection magnitude. That experiment moves up in priority.

### A measurement-validity problem: forced choice by digit

At L9, α=0.2 the *free-form* score is **0.33** while the digit forced-choice score
is **0.05** — below chance — on the same trials. The concept is evidently present
and expressible; the model just cannot map it to a position in a numbered list.

Digit-indexed forced choice is therefore partly measuring format-following, and
at 0.5B that tax is large enough to swamp the signal. Fixed by scoring the option
*words* directly (`WORD_CHOICE`, `identify_word_correct`), with the observer arm
scored the same way so the gap compares like with like. A large divergence
between the digit-scored and word-scored gaps is now itself a diagnostic.

The 1440-trial run above predates that fix, so its identification and gap columns
carry the format tax. Detection AUROC and the null gate do not — those never
involved an indirection.

*Generalisation:* any forced-choice elicitation on a small model needs a
format-competence control. Without one, "cannot introspect" and "cannot follow
the answer format" are the same number.

### Also

- Concept banks stayed non-degenerate at every layer probed (max off-diagonal
  cosine 0.31–0.32), so the sink guard is holding.
- Behavioural KL rises steeply with depth-adjusted α: 0.02 → 2.7 across the
  strengths probed at L9, but only 0.03 → 0.35 at L19. The coherent window really
  does shift with layer, so a fixed α across a layer sweep confounds layer with
  damage — as suspected in the previous entry. Per-layer α calibration to
  constant KL is now required, not optional.
