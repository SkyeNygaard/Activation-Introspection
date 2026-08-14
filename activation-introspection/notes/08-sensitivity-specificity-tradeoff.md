# Introspection training buys sensitivity and pays for it in specificity

Run date: **2026-08-10**

> **CORRECTION, 2026-08-14 — the title of this note overstates what it measured,
> and so does everything downstream of it (notes 29, 31, 32 and 34 inherit the
> framing).**
> The measurements stand; the word "specificity" does not. The random-direction
> condition injects its arbitrary direction into the four demonstrations *and*
> into the query, and the correct label follows the query sign
> ([`run_remap_training.py:493-503`](../scripts/run_remap_training.py)), so those
> rows are a solvable coding task with a right answer — not trials where nothing
> was planted. High trained accuracy there means the model can bind a label to an
> arbitrary demonstrated axis, not that it reports concepts that were absent.
>
> The defensible statement is: **training collapses the advantage that
> concept-derived directions held over arbitrary demonstrated ones.** The
> statement this note's title implies — that a trained monitor answers "did
> something move?" and so fires on anything — requires a condition where the
> query carries no edit and there is no correct label. That condition is written
> (the `none` arm) and **has never been run**. Read notes 29, 31, 32 and 34 with
> this correction attached; their numbers are unaffected, their framing is not.
>
> One thing this file already had right and the rest of the repository did not:
> [`results/README.md`](../results/README.md) records that a prompt-only learner
> scores **0.000** on query-twin pairs. That is the structural null. The "0.25
> coin-flip null" used in the handoff and the READMEs contradicted a number this
> repository had already written down.

## Question

Two questions, asked with one apparatus.

The first was a criticism of [notes/07](07-trained-activation-reporter.md). That
study trained a reporter under a single fixed convention, `+ → Q`, which leaves
the obvious reading that the adapter is a sign probe wired to the output head —
not introspection in any interesting sense.

The second is the limitation named at the end of every other study here: every
edit in this repository is large and out of distribution. Strength 1.0 adds a
direction at the full mean residual norm. Does anything survive at weaker edits?

## Design

Both adapters are trained on episodes byte-identical to the frozen in-context
study — `codebook_icl` is imported read-only for exactly that reason. Same
concepts, carriers, optimizer, and **the same number of gradient steps**. They
differ in one factor:

| adapter | training convention |
|---|---|
| `fixed` | every episode uses `+ → Q` (12 of the 24 cells, shown twice per epoch) |
| `remap` | the convention is re-randomised per episode (all 24 cells, once per epoch) |

Both are then scored against the untrained base model on eight concept directions
and two carriers that neither ever saw, across the full 24-cell enumeration.

### Two nulls fixed by arithmetic

This design rules out two different cheats without measuring either:

| strategy | why it fails | its score |
|---|---|---|
| read only the visible prompt | the two query signs inside an episode give byte-identical prompts with opposite correct labels | **0.000** on query-twin pairs |
| fixed `sign → token` probe | the same hidden state carries opposite correct labels under the two conventions | **0.000** on mapping-flip pairs |

Both are identities, not measured controls. Reported alongside row accuracy
throughout.

### Strength

Training happens at strength 0.5. Evaluation additionally probes 0.25 and 0.15.
A DEV-only calibration on one concept and one carrier fixed those choices before
any held-out bank was touched:

| strength | 1.0 | 0.5 | 0.35 | 0.25 | 0.15 |
|---|---:|---:|---:|---:|---:|
| DEV base accuracy | 0.917 | **0.833** | 0.583 | 0.542 | 0.500 |

Strength 1.0 is at ceiling. A pilot there put base, `fixed` and `remap` all at
1.000 with training loss near zero from the first step: no gradient, so nothing
could move and neither effect could appear. That pilot is why 0.5 was chosen and
frozen. See `results/remap_dev_strength_calibration.json`.

A weaker planted direction is still a planted direction. This is one step along
the naturalness axis, not arrival at it. What it measures is the **detection
floor** of a trained activation monitor.

## Result

Three training seeds, 4,608 rows each, all four gates pass. Every arm holds
format rate 1.00 and label mass 1.000, so nothing here repeats the
[notes/07](07-trained-activation-reporter.md) failure where a reporter scored
well while emitting no labels.

### Finding 1: training extends the detection floor well below the base model's

| strength | base | `fixed` | `remap` |
|---|---:|---:|---:|
| 0.5 *(trained on)* | 0.745 | **1.000** | **1.000** |
| 0.25 | 0.526 | **0.997** | **0.990** |
| 0.15 | **0.500** | **0.863** | **0.790** |

At strength 0.15 the untrained model is at exactly chance — 0.500 row accuracy
and 0.010 on twin pairs. It is blind. Trained only on strength-0.5 examples,
both adapters read that same edit at 0.79–0.86, on concept directions and
carriers withheld from training. Every seed is positive at every strength; the
trained-minus-base contrast is +0.464 to +0.471 at strength 0.25.

Mapping-flip pairs move with it: base 0.438 at strength 0.15 against 0.736–0.806
for the adapters. So this is not a probe reading a sign — the model is reading
the state *and* rebinding an arbitrary convention it is handed in context.

### Finding 2: training destroys semantic selectivity

| direction at strength 0.5 | base | `fixed` | `remap` |
|---|---:|---:|---:|
| concept | 0.745 | 1.000 | 1.000 |
| random, magnitude-matched | **0.513** | **0.955** | **0.913** |

Untrained, the model is at chance on random directions — 0.513 row accuracy,
0.026 on twin pairs — while reading concept directions at 0.745. Its
introspective access is **selective**: it sees internal changes that mean
something and not internal changes that do not.

Training removes that. Random directions go to 0.91–0.96. The trained reporter
has become a generic displacement detector.

### What the two findings say together

> Introspection training buys sensitivity and pays for it in specificity. The
> trained monitor detects internal changes more than three times weaker than the
> base model can see, including ones it never trained on — and stops
> distinguishing changes that mean something from changes that do not.

That is a trade-off a deployed activation monitor would inherit. A monitor
trained this way is more sensitive and less trustworthy about *what* it detected:
asked "is concept X active," it answers a different question, "did something move
at layer 9." An adversary with any displacement at that site gets a positive
report.

> **Extended 2026-08-12 by [`29`](29-can-abstention-recover-selectivity.md), and it
> is worse than this note says.** The obvious repair for a monitor with false
> alarms is to let it abstain when unsure — Anthropic's
> [Introspection Adapters](https://arxiv.org/pdf/2604.16812) names exactly that as
> a candidate fix for the same problem. It does not work here. Dropping the
> least-confident reports **narrows** the gap between real concepts and random
> directions, from 0.059 to 0.013 for `fixed` and 0.099 to 0.019 for `remap`,
> replicated on held-out seeds. The untrained model goes the opposite way — its gap
> nearly doubles, 0.232 to 0.455. So training does not merely make the monitor
> wrong about what it detected; **it makes it most confident exactly where it is
> wrong**, and abstention removes the discrimination that was left rather than
> recovering it.

### The hypothesis this study was built to test, and which failed

Protocol v1 predicted that training under a fixed convention would damage the
model's pre-existing ability to adopt a new convention in context. It does not.
Both adapters reach 1.000 on row, twin and mapping-flip accuracy; `fixed` is if
anything marginally *better* at low strength (0.863 against 0.790 at 0.15, with
overlapping seed ranges). Two v1 seeds falsified it and are retained at
`results/remap_training_v1_*` with `gate set v1` and `all_gates_pass=false`.

The convention used during training does not matter. What is learned is
convention-independent, which is itself the answer to the probe objection: a
model that scores 1.000 on mapping-flip pairs cannot be a fixed sign-to-token
readout, because that scores 0.000 there by construction.

## What does not follow

- Three seeds on one model, one layer, one binary variable. A mean and a range,
  **not** a confidence interval.
- Every edit is still an injected direction. Nothing here shows the model
  reporting a state it computed on its own, which remains the largest gap between
  this work and the project's eventual target.
- The specificity loss is measured against magnitude-matched random directions,
  not against downstream-damage-matched ones.
- "Extends the detection floor" is measured over three strengths on one bank. The
  floor is bounded below 0.15; it is not located.
- `fixed` marginally beating `remap` at strength 0.15 has overlapping seed ranges
  and should not be interpreted.
- No independent human review, no reproduction on other hardware.

## Disclosed deviations

- The strength-1.0 pilot was run and its outcome inspected before 0.5 was frozen.
  It is reported above rather than omitted. It could not have selected an
  outcome, because every arm was at ceiling.
- Protocol v2 was written after v1's hypothesis was falsified. Gates C and D were
  declared before any v2 artifact existed. Gate D is the prospective test of what
  was a post-hoc observation in v1, that training makes random directions
  readable.
- **The analyzer's gates live in code, not in the protocol JSON.** Editing the
  analyzer for v2 silently re-judged the v1 artifact from fail to pass on a first
  run. The gate set is now keyed to the protocol that produced the artifact and
  the summary records which set was applied. v1's verdict is preserved.
- Between the two adapters the runner originally cloned the whole model state
  dict to "restore" base weights. LoRA freezes those weights and `unload` strips
  the adapter without merging, so it restored nothing and cost about 6 GB of
  wired memory, pushing the machine into swap. It is replaced by a weight
  fingerprint that fails the run closed if base weights ever change; a direct
  check confirmed they are bit-identical across train and unload, and peak MPS
  allocation fell from roughly 13–14 GB to 7.2 GB.

## Artifacts

- DEV calibration: `results/remap_dev_strength_calibration.json`;
- frozen v2 protocol: `results/remap_training_protocol_v2.json`, SHA-256
  `f29b479d…be0f0`;
- v2 raw rows, one per seed: `results/remap_training_v2_seed{0,1,2}_raw.jsonl`,
  SHA-256 `0ab9de44…2e97a1`, `ae4ff7ff…dfa210`, `f0005147…fb7e42`;
- pooled summary: `results/remap_training_v2_seeds_summary.json`;
- falsified v1: `results/remap_training_protocol_v1.json` and
  `results/remap_training_v1_seed{0,1}_*`, summary carries `gate set v1`;
- runner and analyzer: `scripts/run_remap_training.py` (SHA-256 `03e8ba2a…6c66`),
  `scripts/analyze_remap_training.py`;
- regenerate with `make remap-training-report`.
