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
