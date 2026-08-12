# Pre-run note: is it content, or just a disturbance?

Written **2026-08-12, before the run.**

## The question this settles

[`13`](13-shared-axis-audit.md) measured that the reporting task in `06`, `07`,
`11` and `12` collapses to a single question — *was this state pushed along one
axis, plus or minus* — and never requires knowing which concept was injected. The
eight concept directions share a common ingredient, all 56 pairs overlap
positively, and the fitted reader is the average concept direction.

That leaves the foundational result of this repository undetermined. The 0.891
shows the model using a causally injected state as an in-context signal. It does
**not** show the model reading *content*, and until now nothing has separated those.

## What I am about to do

Change the two classes from **one concept and its negation** to **two different
concepts**. Change nothing else.

Today the query twins are `+ocean` against `−ocean`. They become `ocean` against
`volcano`. Same four-demonstration interface, same 24-cell enumeration of six
demonstration orders × two label maps × two query states, same carriers, same
byte-identical visible text, same gates.

The property that makes this design worth anything is preserved exactly: query
twins remain byte-identical in visible text while carrying opposite correct
labels, so **a learner using only the input is still pinned at 0.500 by
construction.**

## The arms

| arm | the two classes | why it is here |
|---|---|---|
| `polarity` | `+v` and `−v` | the anchor. Reproduces the published design on these concepts |
| `content` | `v_A` and `v_B`, two concepts | the question |
| `random_pair` | `r_A` and `r_B`, two random directions | **the control that matters** |
| `query_only` | content, demonstrations left clean | pinned at 0.500 by construction |
| `clean` | nothing injected | validity |

**The random pair is the arm that makes this experiment worth running.** "The
model can tell two different injected directions apart" does not require the
directions to mean anything — two random directions are equally distinguishable as
geometry. Without this arm, a positive content result would have two explanations
and would be nearly worthless. With it:

- `content` ≈ `random_pair` → the model discriminates directions, not meanings
- `content` > `random_pair` → something semantic is being read
- both ≈ 0.500 while `polarity` ≈ 0.89 → only the two-poles-of-one-axis structure
  works, and every reporting number here is disturbance detection

## The separation confound, and how it is removed

Opposite poles are further apart than two different concepts. With the injection
normalised to unit length and scaled by the residual magnitude, the distance
between the two classes is `strength × ‖h‖ × ‖unit(a) − unit(b)‖`:

- polarity: `‖unit(v) − unit(−v)‖` = **2.000**
- content: `‖unit(v_A) − unit(v_B)‖` = `sqrt(2 − 2·overlap)` ≈ **1.344** at the
  measured overlap of 0.097

So a naive content arm would be run at two-thirds the separation and any drop
would have two explanations. Every non-polarity arm therefore uses **strength
`2 / ‖unit(a) − unit(b)‖`, computed per pair**, which makes the distance between
the two classes identical across arms by construction. The polarity arm's own
strength is unchanged, since that formula returns exactly 1.0 for opposite poles.

This is the difference between an experiment worth running and one whose negative
result could be explained away afterwards.

## What each outcome means

| Outcome | Reading |
|---|---|
| `content` near `polarity`, and above `random_pair` | The model reads content. The shared axis bounds the interpretation of the old numbers but not the capability, and this is the strongest result in the repository |
| `content` near `polarity`, and `random_pair` equally high | The model discriminates directions regardless of meaning. Honest and useful: it is a real capability, and it is not semantic. This is what `08`'s random-direction result already hints at |
| `content` collapses toward 0.500 while `polarity` holds | Only the opposite-pole structure works. Every reporting number in this repository is disturbance detection, and the branch closes with a clean negative |
| `polarity` itself fails to reproduce | Something is wrong with the harness, not the model. Stop and fix before reading anything else |

There is no outcome I would be embarrassed to report, and no outcome that leaves
the question where it is now.

## What it costs

Inference only. No training. Four arms at 288 episodes plus a 72-episode clean arm
— about 1200 forward passes on the existing 24-cell design. A two-cell smoke runs
first and is disclosed whatever it says.

## What this cannot do

It cannot say anything about naturally computed states; those remain blocked
behind the demonstration-budget question in
[RESEARCH-DIRECTION.md](../../spar-application/RESEARCH-DIRECTION.md). It is one
model, one layer, one set of concepts. And a positive here would be a statement
about *injected* content, which is not the same as content the model computed for
itself.

## Novelty position, stated before the result exists

The live argument in the literature is exactly this one —
[Detecting the Disturbance](https://arxiv.org/abs/2512.12411) and
[Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v1) both
address whether concept-injection results reflect content or generic perturbation
detection, and the second reports gating machinery that fires the same way for
concepts a model detects 97% and 0% of the time.

What appears not to have been run is this comparison with the **visible text held
byte-identical** so an input-only learner is pinned at chance by construction. That
is an extension candidate, not a first demonstration, and per
[LITERATURE-BOUNDARY.md](../../spar-application/LITERATURE-BOUNDARY.md)'s own rule
it needs a proper search against the as-built design before any novelty claim.

---

# Result: it is content, and the disturbance account does not explain it

Run **2026-08-12**. 1224 episodes, 379 seconds. Artifacts:
`results/content_vs_disturbance_v1_raw.jsonl`,
`results/content_vs_disturbance_v1_summary.json`. Runner:
`scripts/run_content_vs_disturbance.py`.

| arm | accuracy | twin-pair | format |
|---|---:|---:|---:|
| `polarity` — `+v` vs `−v` | 0.917 | 0.833 | 1.000 |
| **`content` — two concepts** | **0.899** | **0.799** | 1.000 |
| `random_pair` — two random directions | **0.594** | **0.188** | 1.000 |
| `query_only` | **0.500** | 0.014 | 1.000 |
| `clean` | **0.500** | 0.000 | 1.000 |

All four arms are at **identical separation between the two classes by
construction** — strength 1.000 for opposite poles, 1.444–1.492 for concept pairs,
1.393–1.423 for random pairs, each computed as `2 / ‖unit(a) − unit(b)‖` so the
distance between classes is 2.0 everywhere.

Both structural controls land exactly where the design forces them: `query_only`
at **0.5000** with twin-pair 0.014, and `clean` at **0.5000**. Label format is
1.000 in every arm and mean label mass never drops below 0.997.

## The answer

**The model discriminates two different injected concepts nearly as well as it
discriminates one concept from its own negation — 0.899 against 0.917 — while two
random directions at the same separation sit at 0.594.**

That is the first outcome in the table above: content, not disturbance.

## Every pair, because a mean can hide an outlier

| pair | polarity | content | random | content − random |
|---|---:|---:|---:|---:|
| garden \| camera | 0.931 | 0.889 | 0.569 | **+0.320** |
| train \| banana | 0.889 | 0.889 | 0.778 | **+0.111** |
| eagle \| library | 0.931 | 0.917 | 0.514 | **+0.403** |
| hammer \| island | 0.917 | 0.903 | 0.514 | **+0.389** |

Content beats random on **4 of 4 pairs**, and is within 0.042 of the polarity
anchor on every one. The twin-pair statistic separates them further — content
0.778/0.778/0.833/0.806 against random 0.139/0.556/0.028/0.028 — because a cell
counts only when both query states are read correctly, which a
one-label-regardless strategy cannot do.

`train|banana` is the one soft pair, with random at 0.778. It is the pair where
content's margin is smallest, and it is the reason the per-pair table is reported
rather than only the mean.

## What this does and does not overturn

**It does not overturn [`13`](13-shared-axis-audit.md).** That audit stands: the
*old* task's concept directions share a common ingredient, all 56 pairs overlap
positively, and the fitted probe is the average concept direction. Everything
`13` says about the old bank remains true.

**It bounds what `13` licensed anyone to conclude from it.** The natural next
inference from a degenerate bank — that every reporting number here is disturbance
detection — is now tested directly and is **false**. The new task cannot be solved
by projecting onto one axis, because the two classes are two different concepts
rather than two sides of one, and the model solves it anyway.

**Both effects are real, and that is the honest summary.** Random pairs at 0.594
accuracy and 0.188 twin-pair are *above* chance, so a generic
something-was-pushed component exists, exactly as `13` and the disturbance
literature would predict. Content adds **+0.305 accuracy and +0.611 twin-pair on
top of it.** The generic component is real and small; the content-specific
component is real and large.

## Limits, stated plainly

- **The matched-cost comparison has not been run on this task.** [`11`](11-matched-cost-reader.md)
  showed a four-shot reader dominating the model on the polarity task. Nothing here
  says whether it also dominates on content, and until that runs, **this is not a
  privileged-access result and must not be described as one.** It is a capability
  result: the model reads injected content. Whether it reads it better than a cheap
  outsider is the next question and is currently unanswered.
- The content is **injected**, not naturally computed. The natural-state branch is
  still blocked behind the demonstration-budget question.
- Four concept pairs, one model, one layer, one injection strength, one random
  seed per control direction. Four pairs is a small sample even at 4/4 consistency.
- The polarity arm here uses four concepts, not the eight behind the published
  0.891, so its 0.917 is an internal anchor and not a reproduction of that number.

## Disclosed smoke

A two-cell smoke ran first under the same protocol and is kept at
`results/content_vs_disturbance_smoke_v1_*`. It scored content 1.0, polarity 1.0,
random 0.5, clean 0.5 and query-only 0.5 on **two episodes per arm**, which is
uninformative and is recorded only because the protocol requires disclosing it. No
setting, arm, gate or threshold changed between the smoke and the run above.
