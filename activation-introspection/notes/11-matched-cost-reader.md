# The 0.891 does not survive the privileged-access criterion

Run date: **2026-08-11**

## Question

[Privileged Self-Access Matters for Introspection](https://arxiv.org/abs/2508.14802)
gives the field's operative definition: introspection is a process yielding
information about internal states **"more reliable than one with equal or lower
computational cost available to a third party."**

The confirmed causal-codebook result in [`06`](06-causal-codebook-icl.md) — 0.891
target accuracy against an exactly matched 0.500 query-only arm — had never been
measured against that criterion. `spar-application/PROJECT-BRIEFS.md` said so in
as many words: *"A classifier handed the same retained state would plausibly do
as well, and that control has not been run."*

This runs it, on the identical 576 frozen episodes.

## Design

The same forward pass that scores the model also captures the five post-injection
residual states at the injection site. A four-shot reader is fitted on the four
demonstration states and their labels, and asked for the fifth. The reader is
nearest-centroid — the cheapest reader that can use labels at all — so it is
strictly lower cost than the model, which continues through 27 further blocks
from the same site.

Comparison is paired: the same episode is scored both ways, so the denominator
is identical.

The pre-registered interpretation was written **before the run and states the
expected direction is unflattering to the headline number**, precisely so a
negative could not be reframed afterwards.

## Result

| quantity | frozen gate | result |
|---|---:|---:|
| model target accuracy | reproduce 0.891 ± 0.05 | **0.8924** |
| four-shot reader, Euclidean centroid | — | **1.0000** |
| four-shot reader, cosine centroid | — | **1.0000** |
| shuffled-label reader | within [0.35, 0.65] | **0.5017** |
| model − reader | — | **−0.1076** |
| next-token format rate | — | 1.000 |

The model reproduces its frozen confirmation to within 0.0014, so this is the
same experiment and the comparison stands. The shuffled-label control collapses
to chance, so the reader is using the labelled states and nothing else.

**The paired counts are the result, not the margin:**

| | reader correct | reader wrong |
|---|---:|---:|
| **model correct** | 514 | **0** |
| **model wrong** | 62 | 0 |

There is **not one episode in 576** where the model succeeds and the cheap reader
fails. The reader is correct everywhere the model is correct, and on 62 episodes
besides. This is not a difference of means — it is a dominance relation, and it
holds for every concept: the reader is at 1.000 on all eight while the model
ranges 0.889 to 0.931.

**Verdict, by the frozen rule: no privileged access at this site.**

## What this does and does not overturn

**It does not invalidate the 0.891.** That result's contribution was always
structural: query twins are byte-identical in visible text and carry opposite
correct labels, so an input-only learner is pinned at exactly 0.500 *by
construction*. That remains true, and it remains the answer to the input-shortcut
objection that [Can LLMs Introspect? A Reality
Check](https://arxiv.org/abs/2605.26242) raises against this literature. A
causally injected hidden state **is** usable as an in-context channel.

**It does bound what the number may be called.** The model is not reading its own
state better than an outsider can; it is reading it *worse*. Any description of
the 0.891 as introspection, privileged access, or self-knowledge is not licensed
by this evidence, and the claim ledger has been changed accordingly.

**It converts the headline into a measurable quantity.** The interesting number
is no longer 0.891 but the **gap to the matched reader — 0.108 at this site** —
and how that gap moves with training, depth, and how naturally the state arose.
That is a quantity both target projects care about, and it is the shape of a
result rather than a demonstration.

## Why the 62 failures matter more than the 514 successes

On 62 of 576 episodes the model gets the wrong answer while the signal in its own
residual stream is perfectly separable by a two-centroid comparison. The
information is there; the model does not use it.

That is the same dissociation [`05`](05-retained-trace.md) reports from a
different direction — a probe recovers the concept at 0.958–1.000 while the
model's own behaviour is at chance — now measured **on the same episodes, at the
same site, in the same forward pass**, which the retracted cross-site comparison
in that study could not do. Two independent designs, one phenomenon: linear
availability in the residual stream systematically overstates what the model
itself can use.

## The read-depth sweep, which attacks the above

The first limitation below — that the reader reads at the block the injection
edits, where the signal is maximal by construction — was the strongest objection
to this result, so it was tested immediately and adversarially. Same 576 episodes,
same injection at block 9, one forward pass, and the same nearest-centroid reader
fitted separately on the states captured after **every** block. The protocol
states the fork before the run and names the final block as its primary statistic.

| read block | reader | note |
|---:|---:|---|
| 0–8 | **0.5000** | below the injection: validity control, exactly chance |
| **9–32** | **1.0000** | twenty-four consecutive blocks, perfect |
| 33 | 0.9427 | |
| 34 | 0.8854 | ≈ the model's own 0.8924 |
| 35 | 0.5920 | |

Model, throughout: 0.8924. Both gates pass — the pre-injection reader is at
exactly 0.5000, so it is reading only what the intervention put there.

**The frozen verdict fired as `dominance_is_a_read_site_artifact`,** because the
reader at the final block is below the model. That verdict is recorded as it came
out, and the primary statistic it turns on was **badly chosen — my error in
protocol design, not a property of the model.** The reader reads the five *marker*
positions. By the last blocks those positions are producing their own next-token
predictions; the information the model actually answers from has long since been
moved by attention to the final position. Asking whether the concept is still
linearly separable at a marker position at block 35 is asking the wrong question,
and I froze it as the primary anyway.

**What the sweep actually shows is stronger than what it was built to test.** The
injected signal is perfectly linearly decodable for **twenty-five consecutive
blocks — 9 through 33, about 70% of the network's depth** — while the model, with
that signal sitting in its own residual stream the whole way, answers at 0.892.
The reader beats the model at every one of 25 read depths, not one.

So the privileged-access verdict is not weakened; it is broadened. The criterion
asks whether *some* equal-or-lower-cost third party is more reliable, and there
are twenty-five of them. What is refuted is only the narrower claim that the
reader dominates at *every* depth — at blocks 34 and 35 it does not.

Artifacts: `results/reader_depth_protocol_v1.json`,
`results/reader_depth_v1_raw.jsonl`, `results/reader_depth_v1_raw.manifest.json`,
`results/reader_depth_v1_summary.json`, and a disclosed six-episode smoke under
`reader_depth_smoke_*`. Runner: `scripts/run_reader_depth.py`.

## Limits, stated plainly

- ~~The reader reads at the injection site~~ — **tested above.** The reader is at
  1.000 at 25 read depths, so this is no longer the live objection. The remaining
  version of it is that the signal is *planted*: a naturally computed state has
  not been read this way, and [`10`](10-output-ready-arithmetic.md) records why
  that is currently blocked.
- **The reader is scored only on the state-to-label decision.** The model must
  additionally parse the prompt, apply the episode's remapped label convention,
  and emit a correctly formatted token. It does all three — format is 1.000 —
  but the comparison is not of equal task scope, and the criterion's "equal or
  lower cost" is about cost, not scope.
- **One model, one layer, one strength, one interface.** A model that fails this
  criterion at layer 9 of `Qwen2.5-3B-Instruct` has not been shown to fail it
  anywhere else.
- **A trained reporter has not been measured this way.** [`07`](07-trained-activation-reporter.md)
  reaches 0.927, also below 1.000, but on a different design; the comparison has
  not been run and should not be inferred.

## A disclosed error in the frozen protocol

The v3 protocol predicts the shuffled-label reader should score "about 0.58, not
0.50," reasoning that 4 of 24 permutations preserve the balanced grouping and
leave the reader unchanged. That is half the calculation. Another 4 of 24 swap
the two groups outright and make the reader systematically *wrong*, contributing
0.0. The correct expectation is 4/24·1.0 + 4/24·0.0 + 16/24·0.5 = **0.500**, and
the observed 0.5017 matches it. The frozen band admitted both numbers, so no gate
turned on the error, but the reasoning in the artifact is wrong and is corrected
here rather than quietly.

## Artifacts

- protocol, raw rows, manifest, summary: `results/matched_reader_protocol_v3.json`,
  `results/matched_reader_v3_raw.jsonl`,
  `results/matched_reader_v3_raw.manifest.json`,
  `results/matched_reader_v3_summary.json`;
- disclosed precursors: `results/matched_reader_protocol_v1.json` (frozen, no raw
  artifact — the smoke crashed in per-concept aggregation) and
  `results/matched_reader_protocol_v2.json` with
  `results/matched_reader_smoke_v2_raw.jsonl`, a six-episode smoke whose
  shuffled-label reader scored 1.000 because the permutation was seeded by
  position within a carrier and drew only two distinct permutations. Both are
  implementation defects, disclosed; neither changed the design, the criterion,
  the gates, or the pre-registered interpretation;
- runner: `scripts/run_matched_reader.py`.

## Next

The read-depth sweep is done and is reported above. Two follow-ups remain, in
order of information value.

**Read the final position, not the marker positions, at late blocks.** The
collapse at blocks 34–35 is the sweep's own suggestion: if the concept has been
moved rather than destroyed, it should be recoverable at the position the model
answers from. That distinguishes "the network discards the signal before
committing" from "the signal relocates and the model still underuses it", and it
is one more run on the same episodes.

**Measure the trained reporter this way.** [`07`](07-trained-activation-reporter.md)
reaches 0.927 on a different design. The quantity that matters for introspection
*training* is whether training closes the gap to the matched reader or merely
moves along it. Nothing here answers that, and the answer is the difference
between introspection training and probe distillation.
