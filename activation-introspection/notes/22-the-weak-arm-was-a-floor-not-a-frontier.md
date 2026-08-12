# 22 — The weak arm was a floor, not a frontier

A handoff review ranked "weak-signal privileged-access crossover" as the single
highest-value next experiment, on two legs:

1. the indirect comparison — `08`'s adapters at 0.790–0.863 against `15`'s
   reader at 0.833, at strength 0.15;
2. `15`'s **14 model-only episodes**, described there as "the first in this
   repository" and as the one regime where the reader's dominance breaks.

Leg 2 is an artifact. It cost nothing to check: no GPU, no new run, just the
saved raw rows in `results/matched_reader_content_v1_raw.jsonl`.

## What the 14 episodes are

`15` scored the weak arm by **row**. The frozen protocol in
`scripts/run_report_training.py` says the inference unit is the **twin pair** —
"a pair is correct only when both byte-identical members get their opposite
labels right". Rescoring the same rows at that unit:

| arm | rows: model | rows: reader | pairs: model | pairs: reader | model-only pairs |
|---|---:|---:|---:|---:|---:|
| `content` | 0.899 | 1.000 | 0.799 | 1.000 | 0 |
| `polarity` | 0.917 | 1.000 | 0.833 | 1.000 | 0 |
| `random_polarity` | 0.663 | 1.000 | 0.326 | 1.000 | 0 |
| `polarity_weak` (0.15) | 0.497 | 0.833 | **0.007** | 0.667 | **1** |

At strength 0.15 the model scores **1 twin pair in 144**. The coin-flip null for
twin pairs is 0.25. The model is not at chance; it is an order of magnitude
*below* chance, and 14 model-only episodes collapse to 1 model-only pair.

## Why it is below chance

The per-cell predicted labels say it outright. In **16 of the 24 cells** the
model emits one constant label for all 12 episodes, regardless of which sign was
injected:

| cell | reader ok | model acc | model predictions |
|---|---|---:|---|
| `o0m0q+1` | yes | 1.000 | `Q`×12 |
| `o0m0q-1` | **no** | 0.000 | `Q`×12 |
| `o1m1q+1` | yes | 0.000 | `Q`×12 |
| `o1m1q-1` | yes | 1.000 | `Q`×12 |
| `o5m1q+1` | **no** | 0.000 | `Q`×12 |
| `o5m1q-1` | yes | 1.000 | `Q`×12 |

Each constant-label cell pairs with its twin at 1.000 and 0.000. Row accuracy
averages to 0.4965 — which *reads* as chance, and was reported as chance — while
the model is in fact blind to the edit and falling back on a demonstration-
dependent label preference. A constant-label responder is capped at 0.500 on
rows and pinned at 0.000 on pairs. That is the whole effect.

So every model-only *episode* is a cell where the constant label happened to
match the query sign. None of them is evidence of state sensitivity.

`08` already recorded this floor and I missed it on the first pass: "at strength
0.15 the untrained model is at exactly chance — 0.500 row accuracy **and 0.010 on
twin pairs**. It is blind." My independent recomputation gives 0.007 on a
different bank. The two agree. The claim in `15` was inconsistent with a number
already in the repository.

## A second counting error underneath it

Reader correctness is **constant within a cell** — 24 cells, 12 rows each, zero
cells where it varies. The reader is fit per cell, so its verdict is a
deterministic function of the cell, not of the row. `15`'s weak-arm reader
accuracy of 0.833 is 20 of 24 cells, not 240 of 288 episodes.

Anything computed as if those 288 rows were independent is overstated by roughly
a factor of 12 in effective sample size. On a first pass I quoted an exact
binomial p = 0.0055 for "the model does worse than chance where the reader
fails"; that test treats 48 clustered rows as 48 draws and does not survive the
clustering. The four reader-wrong cells run 0.000, 0.583, 0.583, 0.000. With
n = 4 there is no test worth quoting, and the point estimate is not the finding.
The finding is the constant-label floor, which needs no test.

## What survives

**Leg 1 survives, and is metric-consistent.** `08`'s sweep table is row
accuracy, and `15`'s 0.833 is row accuracy, so 0.790–0.863 against 0.833 compares
like with like. It also cannot be produced by the artifact above: constant
labelling caps row accuracy at 0.500, and the adapters are at 0.79–0.86. `08`'s
mapping-flip pairs move too (base 0.438 against 0.736–0.806), which rules out a
fixed sign→token rule. The trained adapters were reading the weak edit.

**But the comparison has never been made at the protocol's own unit.** `08`
reports twin-pair accuracy at strength 0.5 and only row accuracy across the
strength sweep, so the adapters' twin-pair score at 0.15 is unknown. The
reader's is 0.667. The adapters cannot be rescored — they were never saved.

## Consequence

The reopen condition was "the dominance relation breaks at weak strength." It
does not break; it was never tested at the unit where it could break. What
remains is a genuine gap — one unpaired row-level comparison across two banks —
not a regime where the model showed something the reader missed.

That is a weaker motivation than the handoff assumed, and it changes what the
rerun is for: not chasing an observed crossover, but making a comparison that
has never been made at twin-pair level, with adapters saved so it never has to
be reconstructed indirectly again.

## For whoever runs it

`scripts/run_report_training.py` never saves the adapter — no `save_pretrained`,
no `torch.save`. That is the defect that made this indirect. The script hashes
itself into `source_files_sha256`, so adding adapter-saving changes the frozen
protocol and `freeze_protocol` will refuse to run against `report_training_
protocol_v3.json`. It must ship as a deliberate **v4**, not a quiet edit.

Primary metric for v4 is twin-pair accuracy, reported per training seed. Row
accuracy goes in the appendix, where it can no longer be mistaken for chance.
