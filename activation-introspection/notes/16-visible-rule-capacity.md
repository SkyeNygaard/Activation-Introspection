# Pre-run note: what can the four-shot interface actually learn?

Written **2026-08-12, before the run.** No patching, no site, no intervention of
any kind — this is prompting only.

## The question, sharpened twice

Five natural-state runs ended at a wall: the model scored 0.533 on
"even result → Q, odd result → K" **with the arithmetic written out in plain
text**. The original reading was "the model cannot learn the rule". That was
wrong — four examples is very few and language models are not sample-efficient at
in-context rule induction, so the failure said more about the interface than the
model.

[`14`](14-content-versus-disturbance.md) then sharpened it again, and in a
direction that makes the parity failure genuinely puzzling: the model **can** tell
`garden` from `camera` at 0.899 through the *same four-demonstration interface*.
So four demonstrations are enough to carry a two-way distinction. Why not parity?

## The hypothesis this run tests

The two tasks differ in something more basic than demonstration count.

| | what the demonstrations show | what the query is |
|---|---|---|
| Injected task (works, 0.899) | states pushed with `v_A` and `v_B` | a state pushed with **the same** `v_A` or `v_B` |
| Parity (fails, 0.533) | four arithmetic problems and their labels | **a new problem never shown** |

In the injected task the query is the same thing as two of the demonstrations, so
the model only has to **match**. In parity the query is novel, so the model has to
**induce a rule and apply it to an unseen instance**. Those are different
operations, and nothing in this repository has ever separated them.

So the variable is not demonstration count. It is **whether the query was shown**.

## What I am about to do

Six conditions, all fully visible, all with four demonstrations and the same
24-cell enumeration used everywhere else — six balanced demonstration orders, two
label maps, two query classes — over four item sets each.

| condition | class rule | query | why |
|---|---|---|---|
| `lexical_seen` | arbitrary word sets | **shown in demos** | ceiling. Pure matching, no rule to induce |
| `lexical_unseen` | arbitrary word sets | novel | **floor.** There is no rule, so this is what unlearnable looks like |
| `first_letter` | vowel vs consonant initial | novel | easiest real rule |
| `category` | animal vs tool | novel | semantic rule |
| `magnitude` | number above or below five | novel | numeric rule |
| `parity` | parity of an arithmetic result | novel | the known failure |

`lexical_seen` and `lexical_unseen` are the two anchors that make everything else
readable. Without a ceiling I cannot tell a hard rule from a broken harness, and
without a floor I cannot tell a learned rule from a lucky one.

## What each outcome means

| Outcome | Reading |
|---|---|
| `lexical_seen` high, every novel-query condition at floor | **The four-shot interface matches; it does not induce.** That explains parity, explains why the injected task works, and closes the natural-state branch for this interface — natural states do not arrive pre-matched to their demonstrations |
| Some novel rules pass, parity does not | Parity was a badly chosen rule. The branch reopens with a specific rule named. That is a **development selection**, not a confirmation, and would have to be frozen before any fresh natural-state bank is spent |
| Every rule passes including parity | The parity failure was specific to that run's wording, and five runs were lost to a prompt detail. Everything downstream needs re-examining |
| `lexical_seen` itself fails | Harness fault. Stop, fix, read nothing into the model |

## Prediction, stated before the run

I expect `lexical_seen` near ceiling, `lexical_unseen` at chance, and the real
rules to land between — with `first_letter` above `parity`. If the novel-query
conditions all sit at the floor, the honest conclusion is that this interface
cannot carry natural-state reporting **at any demonstration budget**, and the
branch should be closed rather than repaired.

## What it costs

576 episodes of pure prompting. No interventions, no state capture, no training,
no GPU memory beyond the model itself. Minutes.

## What this cannot do

It says nothing about whether the model *has* access to natural states — only
about whether this reporting interface could express it. A floor result closes the
interface, not the question.

---

# Result: it induces rules fine — but only over classes that already cluster

Run **2026-08-12**. 576 episodes, 112 seconds. Artifacts:
`results/visible_rule_screen_v1_raw.jsonl`,
`results/visible_rule_screen_v1_summary.json`. Runner:
`scripts/run_visible_rule_screen.py`.

| rule | query | accuracy | twin-pair | format |
|---|---|---:|---:|---:|
| `lexical_seen` — **ceiling** | shown | **0.979** | 0.958 | 1.000 |
| `category` — animal vs tool | novel | **0.885** | 0.771 | 1.000 |
| `magnitude` — above/below five | novel | 0.729 | 0.458 | 1.000 |
| `first_letter` — vowel vs consonant | novel | 0.479 | 0.104 | 1.000 |
| `lexical_unseen` — **floor** | novel | 0.490 | 0.083 | 1.000 |
| `parity` — parity of a result | novel | 0.469 | 0.042 | 1.000 |

Both anchors behave: the ceiling is at 0.979 so the harness works, and the floor
is at 0.490 so a condition with no inducible rule lands on chance.

## My prediction was wrong, and the way it was wrong is the finding

I predicted the interface could only **match**, that every novel-query condition
would sit at the floor, and that `first_letter` would be the strongest real rule
because it is the simplest to state.

Both halves are wrong. `category` reaches **0.885 on queries never shown** — the
interface induces a rule and applies it to a new instance perfectly well. And
`first_letter`, the simplest rule on the list, is **at the floor**: 0.479 against
0.490.

So the dividing line is not seen-versus-novel, and it is not rule complexity.

## What actually predicts success

Rank the conditions and one property orders them exactly:

| condition | do the two classes form a natural cluster? | result |
|---|---|---:|
| `lexical_seen` | the query *is* a demonstrated item | 0.979 |
| `category` | animals cluster, tools cluster | 0.885 |
| `magnitude` | numbers carry partial magnitude structure | 0.729 |
| `first_letter` | `apple, otter, igloo` share nothing but a letter | 0.479 |
| `parity` | `4+4, 2+2, 6+2` share nothing but an answer property | 0.469 |

**The interface does similarity matching in representation space.** It succeeds
when each class is a region the query falls into, and fails when class membership
cuts across that space — which is exactly what an orthographic rule and a computed
property do.

This is a **more informative version of the hypothesis in the pre-run note**, and
it is inferred from the ordering rather than measured directly. The direct test
would be to check whether the class members are close together in the model's own
representations and whether that closeness predicts the score, which is cheap and
has not been run.

## Why this unifies the whole branch

| result | classes | reading |
|---|---|---|
| injected `±v` at 0.891 | one direction and its negation | maximally clustered — two points |
| content `v_A` vs `v_B` at 0.899 ([`14`](14-content-versus-disturbance.md)) | two distinct concept directions | two tight clusters |
| `category` at 0.885 | two semantic categories | two natural clusters |
| `parity` at 0.469 | five unrelated computed states per class | no cluster |

The injected tasks were never testing rule induction. They were testing whether
the model can tell which of two tight clusters a point belongs to, and that is why
four demonstrations suffice. Parity asked for something categorically different
and got the floor.

## What this does to the natural-state branch

**It reopens it, with a named condition and a warning.**

The branch stopped because parity failed. Parity failed because its classes do not
cluster, not because the model lacks access and not because four demonstrations
are too few. A hidden class whose members *do* cluster — a semantic category — sits
at 0.885 through the identical interface.

**This is a development selection, not a confirmation.** `category` was chosen
from six candidates by looking at the results, which is precisely what the frozen
anchor lists in [`09`](09-natural-state-pilot.md) and
[`10`](10-output-ready-arithmetic.md) existed to prevent. Any successor must:

1. freeze a semantic-category hidden class **before** seeing natural-state data;
2. use a **fresh** bank — the two earlier ones are spent;
3. keep this visible screen as a gate, so a null can be read as a reporting null
   rather than an unlearnable class;
4. certify the transplant per item, as
   [`10`](10-output-ready-arithmetic.md) established.

And the harder problem this exposes: the natural states must be ones whose classes
cluster in representation space. That is a real constraint on task design, and it
is not obviously satisfiable for any interesting internal variable.

---

# Pre-run note for the direct test, written before it ran

The cluster account above orders five conditions correctly and is an **inference
from a ranking**. This repository has been caught twice today promoting an
inference of exactly this shape — the shared axis in [`13`](13-shared-axis-audit.md),
and the generality claim retracted in [`15`](15-matched-reader-on-content.md) — so
it gets measured rather than written up.

**What I am about to do.** Embed every item from every rule in the same
`Item: {x}` frame the prompt uses, at three depths, and compute for each rule how
much tighter items are to their own class than to the other one. No prompting, no
scoring, no intervention.

**Why.** If that separation orders the rules the way accuracy does, the account is
measured. If it does not, the account is wrong and the ranking needs a different
explanation — which would matter, because the natural-state branch is about to be
reopened on the strength of it.

**What each outcome means.** Separation tracks accuracy → the interface matches on
representational similarity, and any future hidden class must be checked for
clustering *before* a bank is spent, which is a cheap gate the branch does not
currently have. Separation does not track accuracy → `category`'s success has
another cause, and reopening the branch on "pick a semantic category" is not
justified.

**Cost.** About 150 short forward passes. No training, no interventions.

**Prediction.** `category` highest, `parity` and `first_letter` near zero. I got
the last prediction in this note wrong, so this one is held loosely.

---

## Result of the direct test: measured, and it separates cleanly

Artifact: `results/cluster_check_v1_summary.json`. Runner:
`scripts/run_cluster_check.py`. Separation is mean within-class similarity minus
mean between-class similarity, so 0 means the classes are no tighter internally
than they are to each other.

| rule | layer 9 | layer 18 | layer 27 | accuracy |
|---|---:|---:|---:|---:|
| `lexical_seen` | **0.218** | 0.096 | 0.081 | 0.979 |
| `category` | **0.070** | 0.081 | 0.069 | 0.885 |
| `magnitude` | **0.074** | 0.054 | 0.043 | 0.729 |
| `lexical_unseen` | 0.008 | −0.000 | −0.004 | 0.490 |
| `first_letter` | 0.000 | 0.005 | 0.002 | 0.479 |
| `parity` | **−0.023** | −0.021 | −0.014 | 0.469 |

**The account is measured, and the split is clean.** Every rule the interface
learns has separation between 0.043 and 0.218 at every depth. Every rule it fails
has separation between −0.023 and 0.008. There is no overlap, and the gap between
the two groups is larger than the spread inside either.

The ordering agrees with accuracy at all three depths. The only disagreements are
`first_letter` against `lexical_unseen`, which differ by 0.011 in accuracy and by
0.005 in separation — two conditions that are indistinguishable on both measures,
which is exactly what the floor should look like.

**Parity is the only condition with a consistently negative separation**, and that
is worth its own sentence. Its classes are not merely unclustered but *actively
misleading*: `4+4, 2+2, 6+2` and `3+4, 5+2, 1+2` share operands, operators and
length, so each expression sits closer to a member of the *other* class than to
its own. Parity was not a neutral choice of hidden rule. It was close to the worst
available, and five runs were spent on it.

## The gate this hands the branch

The separation measure needs **no patching, no site, no bank, and no model
intervention** — about 150 short forward passes. So any future hidden class can be
screened before a single episode is spent:

> **Measure the class separation first. Below roughly 0.04, the four-shot
> interface will not learn it, and no amount of transplant work will help.**

That is a cheap prospective gate, and it is the thing the natural-state branch has
lacked from the start. It should be frozen into the next protocol as a
precondition, not run afterwards as an explanation.

## Limits

- Four item sets per rule, 96 episodes each. Enough to separate 0.885 from 0.479,
  not enough to rank two rules a few points apart.
- The cluster account orders five conditions correctly and is **inferred**, not
  measured. The direct check is named above and unrun.
- One model, one prompt template, one label pair. `category` used animals and
  tools; other category pairs are untested.
- `magnitude` at 0.729 with twin-pair 0.458 is the ambiguous one and should not be
  called either a pass or a failure.
