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

---

## 2026-08-01 (later still) — Power. The underpowered version was wrong.

### What was wrong

The transfer probe above ran at n_train=48 (6 examples per class over an 896-dim
residual, sample:dimension = 0.007) and n_test=80 (95% CI half-width ±0.110 at
p=0.5). Its null shuffled *test* labels, which only asks "is this above chance",
not the stronger "can this pipeline manufacture signal from noise at this sample
size".

Fixes: 40 natural templates (n_train=320, 40/class), n_test=400 (±0.049),
**GroupKFold on template id** so the within-natural estimate must generalise to
sentence frames never seen in training, and a **permuted-label null** that
retrains the whole pipeline on shuffled training labels, averaged over 5
permutations (null n=2000).

### It changed the answer

`Qwen2.5-0.5B-Instruct`, injection L9, α=0.2, 8 concepts:

| probe layer | train (in-sample) | within-natural (grouped) | **transfer** | permuted null | self-report |
|---|---|---|---|---|---|
| 11 | 1.000 | 0.650 [0.597, 0.703] | **0.125 [0.095, 0.158]** | 0.150 [0.135, 0.166] | 0.125 |
| 16 | 1.000 | 0.700 [0.650, 0.750] | **0.700 [0.655, 0.745]** | 0.155 [0.140, 0.171] | 0.125 |
| 23 | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 0.100 [0.087, 0.113] | 0.125 |

**Layer 11 flipped.** Underpowered it read 0.200 against a 0.150 null — I would
have written up "partial transfer at early layers". Properly powered it is
**0.125 against a 0.150 null**: at or below chance. There is no transfer two
layers after the injection.

That is the whole reason to run the permuted null and the larger n. The
qualitative conclusion changed, not just the error bar.

### What the properly powered result says

The injected state does not start out resembling the model's own representation
of the concept; it *becomes* one with depth. Transfer is at null at L11, reaches
the probe's own within-natural ceiling at L16 (0.700 vs 0.700 — the injected
states are as decodable as genuinely reading about the concept), and is complete
by L23.

The in-sample/grouped-CV split is worth keeping visible: train accuracy is 1.000
at every layer while grouped CV is 0.650–0.700 at L11 and L16. The probe *is*
overfitting; grouped CV is what makes the number mean anything. At L23 they
coincide at 1.000, i.e. the concepts are genuinely linearly separable there.

### The caveat I would put in any writeup

**L23 is nearly output space.** At the final layer, "ocean" natural text and an
"ocean" injection both point toward ocean tokens, so transfer=1.000 there is
close to the token-promotion confound in another guise. It should not be the
headline.

**L16 is the load-bearing number**: mid-depth, well away from the unembedding,
transfer (0.700) matches the probe's own ceiling (0.700) against a 0.155 null,
while the model self-reports at chance. That is the honest statement of the
dissociation.

And the denominator is still contaminated — self-report is digit-indexed forced
choice, which this model fails independently. So "0.700 vs 0.125" overstates the
access gap by an unknown amount. The design is sound; the self-report instrument
is not, and fixing it is the next job.

### Scale ladder, both models (n=96 per model per arm, ±0.10)

| | 0.5B | 1.5B |
|---|---|---|
| detection AUROC, concept | 0.408 [0.324, 0.491] | 0.656 [0.582, 0.733] |
| detection AUROC, **shuffled null** | 0.490 [0.408, 0.582] | **0.740 [0.667, 0.812]** |
| identification (digit) | 0.094 [0.042, 0.156] | 0.167 [0.094, 0.240] |
| observer | 0.177 [0.104, 0.260] | 0.396 [0.302, 0.500] |
| gap, paired | −0.083 [−0.167, 0.000] | **−0.229 [−0.344, −0.125]** |
| token promotion | 0.990 | 1.000 |

At 1.5B the **null arm detects better than the concept arm** (0.740 vs 0.656).
Detection is tracking perturbation, not content — an independent replication of
the published global-logit-shift result, reached with a matched-norm shuffled
control instead of their factually-false-questions control. The validity gate
marks these cells INVALID automatically.

The scale trend is the interesting part: from 0.5B to 1.5B the **observer** more
than doubles (0.177 → 0.396) while the introspector barely moves (0.094 → 0.167),
so the gap gets *more* negative. Behavioural inference scales with capability;
introspective report does not, at these sizes.

n=96 per cell gives ±0.10, which is enough for the sign of these effects but not
for fine comparisons. The 1.5B gap CI excludes zero comfortably; the 0.5B one
touches it.

---

## 2026-08-01 — Full-depth profile, and the position after reading the field

### The literature position (see `notes/00-literature.md`)

The confounds I characterised are published: binary detection is a global logit
shift (arXiv 2512.12411, r=0.999 against factually-false controls); identification
confabulates while detection succeeds (arXiv 2603.05414); the introspective
circuit sits at ~70% depth (arXiv 2603.21396).

The **gap** is that both papers training models to introspect state they do not
compare probe decodability against verbalization:

- *Introspection Fine-Tuning* (arXiv 2607.14111) — Llama-1B 9.6% → 60.6% on
  sentence localization, peaks "at optimal layer/strength configurations", and
  "does not employ linear probes to compare what activations encode versus what
  models report".
- *Introspection Adapters* (arXiv 2604.16812) — 89% verbalization on AuditBench,
  no probe-vs-verbalization comparison, open question stated as *why* they
  generalize.

So the quantity nobody has measured is **how much concept information is linearly
present before any introspection training**. That is the profile below, and it
licenses a falsifiable prediction: IFT gains should track pre-training
decodability layer by layer.

### The profile (Qwen2.5-0.5B, inject L8, α=0.2, n_train=320, n_test=400)

Probe trained only on natural text; permuted-label null; self-report flat at
0.125 (chance) at every layer, since it is one measurement of the model.

| layer | depth | within-natural | transfer | null |
|---|---|---|---|---|
| 9 | 38% | 0.713 | 0.625 | 0.067 |
| 11 | 46% | 0.650 | 0.525 | 0.158 |
| 13 | 54% | 0.650 | 0.500 | 0.050 |
| 14 | 58% | 0.769 | 0.750 | 0.167 |
| 15 | 62% | 0.750 | 0.775 | 0.158 |
| 16 | 67% | 0.700 | 0.775 | 0.083 |
| **17** | **71%** | 0.681 | **0.875** | 0.200 |
| 18 | 75% | 0.691 | 0.825 | 0.100 |
| 20 | 83% | 0.934 | 1.000 | 0.108 |
| 23 | 96% | 1.000 | 1.000 | 0.100 |

Transfer beats its null at every layer from L9 on. The shape has a clear stable
high band at **58–75% depth (L14–18)**, peaking at L17 = 71%, which brackets the
~70% introspective circuit Macar et al. locate in two much larger models. That
convergence across a 0.5B model and 27B/235B models is worth noting and worth not
overreading — it is one model here.

### Two things that keep this honest

**Transfer exceeds within-natural at L17** (0.875 vs 0.681). The injected state is
*more* linearly separable than natural concept text, not merely as separable. That
is expected — a contrast-derived steering vector is a purer version of the
direction than any natural sentence — but it means "the injection induces a
genuinely concept-like state" is too strong. The defensible statement is: a
decision boundary fit on natural text assigns injected states to the correct
concept, and injected states sit further from that boundary than natural examples
do.

**L20+ is near output space** and its 1.000 should not be the headline; at the
final layers "ocean" text and an "ocean" injection both point at ocean tokens,
which is token promotion in another guise. **L14–18 is the load-bearing band.**

### Instability near the injection site

An earlier run injecting at L9 and probing at L11 gave transfer 0.125 (at null).
This run injects at L8 and probes L11 at 0.525. A one-layer shift in the injection
site — which also moves the layer the concept bank is built at — changes the
nearby profile substantially. Downstream of ~L14 the profile is stable across both
configurations (0.75–0.88 vs 0.70–0.78).

So: the band from 58% depth onward is a property of the model; the first few
layers after injection are a property of the configuration. Any claim about early
layers needs the injection site swept, not fixed.

### Standing limitation, unchanged and important

Self-report is digit-indexed forced choice, which this model fails independently
of introspection. The transfer-vs-self-report spread therefore overstates the
access gap by an unknown amount. The profile is the contribution; the gap
magnitude is not quotable until the self-report instrument is fixed.

### Next

1. Sweep the injection site, to separate model structure from configuration.
2. Run the profile on 1.5B and 3B: does the high band stay at 58–75% depth?
3. Cross-lingual self-report, for an honest denominator.

---

## 2026-08-01 — Prediction falsified. The real variable is the remaining compute budget.

### The test

I predicted that pre-training transfer-probe decodability would forecast where
introspection training generalizes. Tested it directly: LoRA fine-tune
Qwen2.5-0.5B to answer the identification question with a concept injected at ONE
layer, then measure accuracy at every other layer. Training and evaluation use
**disjoint prompt paraphrases** (variants 0–2 vs 3–4), so held-out layers are
scored on wordings never trained on. Digit-scored, so not circular. 144 training
examples, 2 epochs, base model frozen.

### The prediction was wrong, with the opposite sign

r = **−0.774** (train L8) and **−0.640** (train L16) between pre-training probe
transfer and post-IFT accuracy. Decodability does *not* predict where training
generalizes — it anti-predicts it.

### What actually governs it

Two training runs, L8 and L16, evaluated on both sides of the trained layer:

| injection layer | layers remaining | trained @ L8 | trained @ L16 | probe transfer |
|---|---|---|---|---|
| 2 | 21 | – | 0.938 | – |
| 8 | 15 | 1.000 | 1.000 | – |
| 14 | 9 | 1.000 | 1.000 | 0.750 |
| 17 | 6 | 0.944 | 1.000 | 0.875 |
| 19 | 4 | 0.812 | 1.000 | 0.600 |
| **20** | **3** | 0.525 | 1.000 | **1.000** |
| **21** | **2** | 0.125 | 0.550 | **1.000** |
| **22** | **1** | 0.125 | 0.194 | **1.000** |
| **23** | **0** | 0.125 | 0.125 | **1.000** |

Pooled over both runs (n=37 layer evaluations):

- **≥4 layers remaining downstream: mean post-IFT accuracy 0.957** (n=29)
- **≤3 layers remaining: mean post-IFT accuracy 0.346** (n=8)

It is not distance from the trained layer. Training at L16 generalizes *downward*
fourteen layers to L2 at 0.938, while failing *upward* at just +5 (L21, 0.550).
An effect that reaches 14 layers in one direction and 5 in the other is not a
distance effect. The failure boundary sits at the same absolute place in both
runs — the last three layers — regardless of where training happened.

### The mechanism, and the dissociation that makes it interesting

The model needs a minimum amount of remaining computation to convert an injected
signal into an answer token. Below roughly three or four remaining blocks it
cannot, and no amount of fine-tuning fixes that.

The sharp part: **this failure occurs exactly where linear decodability is
maximal**. At L21–23 an external probe trained on natural text reads the injected
concept at 1.000, while the fine-tuned model reports at chance. The information
is present, linearly available, and unreportable.

So the two quantities come apart completely:

> **Decodability by an external probe is not usability by the model's own forward
> pass.** A probe may read the final residual directly; the model must route that
> content through its remaining layers into a token choice, and near the output
> there are no remaining layers to route through.

This is why my earlier framing — "decodability bounds verbalization headroom" —
was wrong. Decodability is necessary but nowhere near sufficient; the binding
constraint at late layers is compute, not representation.

### What this says about IFT's open question

*Introspection Fine-Tuning* (arXiv 2607.14111) lists "mechanisms underlying the
layer-agnostic generalization effect" as unresolved. This gives a candidate
answer:

> IFT generalizes across layers not because the learned report is layer-agnostic,
> but because a **remaining-compute budget** governs it. Generalization holds
> wherever enough layers of computation remain downstream of the injection, and
> fails otherwise, largely independent of which layer was trained on.

Two practical consequences, both directly actionable and neither requiring the
training to be run:

1. **Train at mid-depth.** Training at L16 covers L2–L20 (19 layers); training at
   L8 covers L8–L19 (12 layers). Mid-depth training generalizes over strictly more
   of the model, because it is not the distance that matters.
2. **Do not expect introspection training to work on the last few layers**, and do
   not read failure there as absence of the representation — the representation is
   maximally decodable exactly there.

### Limits, stated plainly

One model (Qwen2.5-0.5B), one concept bank, one strength, LoRA rather than full
fine-tuning, and a small training set that drives loss to ~1e-4 (the adapter
certainly memorises the training layer; the claim rests on held-out layers and
held-out paraphrases). The "3–4 layer" threshold is a number from a 24-layer
model and should be expected to scale with depth rather than transfer as a
constant.

The falsification is solid regardless: whatever governs layer generalization, it
is not pre-training decodability, because the two anti-correlate.

### Next

1. Replicate on 1.5B: does the boundary sit at a constant *number* of remaining
   layers, or a constant *fraction* of depth? That distinguishes a fixed compute
   requirement from a proportional one.
2. Vary injection strength at fixed remaining depth — if the boundary is compute,
   a stronger injection should not move it.
