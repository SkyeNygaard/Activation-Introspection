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

---

## 2026-08-01 — The word-scored metric was circular. Control added.

Fixing the digit-format tax from the previous entry produced a spectacular
result, and the spectacular result was an artifact. This entry is the whole
story, because the artifact is the interesting part.

### What it looked like

Scoring identification over the concept *words* instead of digit indices, on
`Qwen2.5-0.5B-Instruct` at L14, α=0.2, 8 concepts × 3 seeds:

| metric | accuracy (chance 0.125) |
|---|---|
| word-scored identification | **1.000** |
| digit-scored identification | 0.125 |
| observer, word-scored | 0.500 |

Perfect 8-way identification from a 0.5B model, with a huge positive gap over the
observer. Taken at face value: strong evidence of privileged access, at a scale
nobody expects it.

### What it actually was

Concept vectors here are contrast directions — built as the difference in
residual stream between prompts containing the concept and matched prompts that
do not. Such a direction *is*, approximately, the direction that raises the
concept's own token. Injecting it raises P(" ocean") mechanically, whether or not
the model has any access to its own state.

So scoring the concept words while injected can recover the answer with no
introspection at all. It reads the steering vector's construction back out.

**The control.** Score the same option words after a neutral prompt that asks
**no question**:

| | accuracy |
|---|---|
| word-scored identification, question asked | 1.000 |
| **token promotion, no question asked** | **1.000** |
| token promotion on the shuffled control | 0.167 |

Asking the question adds exactly nothing. And the shuffled arm sits at chance, so
the control is specific to the real concept direction rather than trivially
returning true — the promotion effect is real, and it is the entire result.

`token_promotion_correct` is now a recorded field, and `analysis.headline`
refuses to report a word-scored gap at all when it fires.

### Standing conclusion at 0.5B

No evidence of introspective access, on the metric that survives its controls:

- digit-scored identification **0.125**, exactly chance
- detection AUROC **0.446** [0.403, 0.489], below chance, with the shuffled null
  at 0.484 [0.436, 0.529]
- paired gap **−0.072** [−0.117, −0.031] — the observer does *better*
- the one metric showing a positive effect is circular

### The generalisable point

This is the second time in this repo that a measurement failed in the direction
of the hypothesis, and both times the failure was silent. The pattern:

> If the elicitation is scored on tokens the intervention directly promotes, the
> experiment cannot distinguish introspection from the steering vector's own
> construction.

That is not specific to my setup. It applies to any concept-injection study that
asks the model to *name* the injected concept and scores the name — which is the
natural and common design. The cheap fix is one extra forward pass per trial: ask
nothing, score the same options, and report it beside the headline.

Three ways to get a non-circular identification metric, in rough order of cost:

1. **Digit or letter indices** — no lexical overlap with the injected direction.
   Cheap, but taxes format-following (0.5B pays that tax heavily).
2. **Paraphrase options** — "a large body of salt water" rather than "ocean".
   Reduces but does not eliminate promotion, since the description shares
   vocabulary with the concept.
3. **Cross-lingual options** — score the concept in a language the vector was not
   built in. Cleanest separation; untested here.

The honest summary of this artifact for a writeup: *the effect size of the
confound (1.000) is larger than any plausible real effect, so a study that omits
this control is not weak evidence of introspection — it is no evidence at all.*

### Next

1. Finish the 0.5B → 1.5B ladder with the control recorded throughout.
2. Matched-KL comparison on the digit-scored arm, which is the only comparison
   both non-circular and not confounded by injection damage.
3. Paraphrase and cross-lingual option sets, to recover format-tax-free
   identification without reintroducing circularity.

---

## 2026-08-01 (later) — Literature check first. Then a real dissociation.

### The novelty check I should have run before writing up anything

Searched the 2026 literature before building a portfolio around the previous two
entries. **Neither finding is novel**, and one of them was my own bug:

| my "finding" | prior work |
|---|---|
| detection has a yes-bias, so use AUROC | *Detecting the Disturbance* (arXiv 2512.12411) — same effect, **better control**: identical injection applied to factually-false questions ("Can humans breathe underwater?") gives r=0.999 correlation with the detection logit shift. Purely mechanical, cleanly demonstrated. |
| identification is at chance / confabulates | *Emergent Introspection in AI is Content-Agnostic* (arXiv 2603.05414) — models detect anomalies but confabulate content toward common concrete nouns. |
| word-scored identification is circular | Not a contribution. Lindsey's design **removes the steering vector before querying the model**, which avoids the confound by construction. I injected during the answer. Self-inflicted. |
| adaptive attackers defeat monitors (repo B) | *Adaptive Attacks on Trusted Monitors* (arXiv 2510.09462); *CIAware-Bench* (arXiv 2606.11063) benchmarks whether models notice a control intervention. |

Lesson worth keeping: I spent two entries characterising confounds that were
already characterised. The check costs twenty minutes and should come before the
experiment, not after the writeup.

### The gap that survived the check

Two published results are in tension:

- binary detection is *entirely* confounded (global logit shift), yet
- identification of **which sentence** was perturbed reaches ~88% against 10%
  chance, while identification of **what** was injected confabulates.

So models localise perturbations they cannot name. Nobody I found tests whether
the content is *absent* from the state, or *present and unreported*. That
distinction decides whether verbalization training has anything to train on.

### Two failed designs, same trap

1. **Raw linear probe on injected activations.** Probe accuracy 1.000 at layers
   11, 16 and 23 (injection at 9), null 0.113. Meaningless: the residual stream is
   additive, the injected direction propagates forward nearly verbatim, and the
   probe recovers what I put there. Probing "downstream" does not help — hence
   1.000 at the *last* layer.
2. **Ablate the injected direction, then probe.** Still 1.000. Projecting out one
   rank-1 direction does not remove its *downstream image*, which is a
   deterministic function of the same injection.

Both are the circularity trap in new clothing. Anything you inject, you can
decode.

### The design that escapes it: cross-distribution transfer

Train the probe **only on natural text** mentioning each concept — no injection
anywhere in the training set — then test it on injected trials. The probe cannot
recover "the vector we added" because it never saw one. It can only succeed if
injection moves the model into the same region of representation space that
genuinely processing the concept does.

`Qwen2.5-0.5B-Instruct`, injection at L9, α=0.2, 8 concepts, n=80 injected trials,
48 natural training examples:

| probe layer | within-natural (probe works?) | **transfer to injected** | shuffled null | model self-report |
|---|---|---|---|---|
| 11 | 0.062 [0.000, 0.146] | 0.200 [0.113, 0.287] | 0.150 | 0.125 |
| 16 | 0.188 [0.083, 0.292] | 0.525 [0.412, 0.637] | 0.113 | 0.125 |
| 23 | 0.958 [0.896, 1.000] | **1.000 [1.000, 1.000]** | 0.138 | 0.125 |

The layer structure is the credibility check. Where the probe cannot separate
concepts even in natural text (L11), nothing transfers — as it must not. Where it
can (L23), transfer is perfect. A vector injected at layer 9 produces, fourteen
layers downstream, a state that a classifier trained purely on ordinary sentences
assigns to the correct concept every time.

**The dissociation:** concept identity is fully linearly decodable from the state
the model answers from, while the model's own forced-choice answer sits at chance.

### What this does and does not license

It does **not** show the model "knows" what was injected. Linear decodability is a
statement about the representation, not about access.

It does bound the target for verbalization training: at this scale the content is
present in a linearly-readable form, so a training signal for verbalizing it has
something to attach to. If the transfer probe had come back at chance,
verbalization training would have had nothing to learn.

**The number is inflated and I should not quote 0.875 as the gap.** Self-report
here is digit-indexed forced choice, which this model fails independently of
introspection (documented two entries up: free-form 0.33 where digits scored
0.05). So part of the 1.000-vs-0.125 spread is format incompetence, not access
failure. The design is sound; the self-report instrument is not.

That is now the concrete open problem, and it is the one worth solving next: a
self-report measure that is neither circular (must not score tokens the injection
promotes) nor format-taxed (must not require concept→digit indirection).
Cross-lingual options are the most promising candidate — score the concept in a
language the steering vector was not built in.

### Next

1. Cross-lingual self-report, to get an honest denominator for the dissociation.
2. Transfer probe across the scale ladder: does the dissociation narrow with size?
   That is the question the whole line is for.
