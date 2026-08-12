# 24 — Is the held-out failure the interface? (pre-run note)

Written before the run. Nothing here was edited after seeing results.

## The question

[`23`](23-held-out-semantic-generalization.md) found the model cannot place an
unseen exemplar into a demonstrated category: 0.083 on twin pairs against a
four-shot reader's 0.986 on the identical states, at the constant-label floor.

That was measured through **one wording**. And this repository has already been
burned by exactly that. [`21`](21-is-the-channel-narrow-or-was-i.md) reported the
model knew five times more than it said — then found the prompt had literally
instructed it not to say anything, and that changing the wording alone moved the
number from 0.292 to 0.708.

So before `23`'s negative is treated as a fact about the model, it has to survive
a serious attempt to elicit the ability.

There is a second reason to run this now and not later. The handoff's training
experiment is supposed to show what training buys. `21`'s lesson is that the
baseline must be elicitation-optimized *first*, or training gets credit for what
better wording would have produced anyway. Running the retrain before this would
repeat the mistake the repo has already diagnosed once.

## What I am about to do

Vary **only the instruction text**. Same exemplars, same injection sites, same
strength, same episodes, same readout, same scoring. Five wordings:

1. **`baseline`** — `23`'s exact wording. The anchor. It must reproduce `23`'s
   0.083 held-out and 0.521 same-exemplar, or the run is not comparable.
2. **`two_groups`** — says outright that the demonstrations fall into two groups
   and the query belongs to one of them.
3. **`generalize`** — says the query is a *new* member, not a repeat, and asks
   which group it belongs with. This names the exact operation `23` found missing.
4. **`introspect`** — directs attention to the model's own internal state rather
   than the text, which is identical everywhere.
5. **`eliminate`** — frames the two labels as exhaustive and asks for the better
   match rather than a recall.

Two arms carried from `23`: `same_exemplar` and `heldout_semantic`. The anchor
matters here because a wording could lift *everything*, which is a different
finding from lifting held-out generalization specifically.

The readout is unchanged — the next token after `Label:` must be `Q` or `K` — so
every number is directly comparable to `23`. What this cannot test is chain of
thought, which needs a generation harness and a different scoring rule. Stated as
a limit, not smuggled in.

## Development and confirmation, split before the run

`birds_buildings` is **development**: the best family is chosen there.
`body_weather` is **confirmation**: that family is reported there without any
further choice. Picking a winner across five wordings and reporting its best
number would be selection on the outcome, which is how `13` and `15` both went
wrong.

Three exemplar draws, one carrier. `23` showed the variance that matters is
between exemplars (`penguin`/`castle` 0.833 against `spine`/`monsoon` 0.250) and
not between carriers, so the rotation budget goes to exemplars.

## What each outcome would mean

**No family lifts held-out above the floor.** `23`'s negative hardens: the failure
is not that the question was asked badly. Combined with the reader at 0.986 on the
same states, that is a clean statement that the model does not use category
structure that is demonstrably present and trivially extractable.

**A family lifts held-out on development and it replicates on confirmation.**
`23`'s negative was an interface artifact and must be withdrawn. This is the
outcome that would most change what I believe, and it is why the run is worth
its cost.

**A family lifts held-out on development but not on confirmation.** Wording
interacts with the specific categories; there is no stable elicitation win. Report
as instability, not as a positive.

**A family lifts the anchor as well as held-out, by similar amounts.** The wording
improved general compliance with the task, not category generalization. The gap
between the two arms is the quantity of interest, not either level.

**The baseline does not reproduce `23`.** Instrument failure. Fix before reading
anything else, exactly as the anchor check in `23` was chased before its result.

## Kill rule

If no wording clears the coin-flip null of 0.25 on held-out in development, stop
varying the elicitation for this interface. The next question then becomes whether
training changes it, not whether a sixth prompt would.

## Cost

Two category pairs, one carrier, 24 cells, three draws, five wordings, two arms:
1440 episodes, about eight minutes at `23`'s measured rate, plus model load. No
training. The frozen episode machinery is subclassed, not modified, so no protocol
hash changes.

---

# Result: the wording works, and it buys nothing at all

Run **2026-08-12**. 1440 episodes. Artifacts:
`results/heldout_elicitation_v1_raw.jsonl`,
`results/heldout_elicitation_v1_summary.json`. Runner:
`scripts/run_heldout_elicitation.py`.

Each cell is 36 twin pairs. The null is 0.25 — an unbiased guesser gets both
members of a twin right a quarter of the time. `p` is one-sided binomial.

| | wording | anchor | **held-out** | k/n | p vs 0.25 |
|---|---|---:|---:|---:|---:|
| **dev** | `baseline` | 0.694 | 0.028 | 1/36 | 1.000 |
| | `two_groups` | 0.611 | **0.167** | 6/36 | 0.917 |
| | `generalize` | 0.861 | 0.139 | 5/36 | 0.966 |
| | `introspect` | 0.833 | 0.111 | 4/36 | 0.989 |
| | `eliminate` | 0.917 | 0.056 | 2/36 | 1.000 |
| **confirm** | `baseline` | 0.500 | 0.167 | 6/36 | 0.917 |
| | `two_groups` | 0.583 | 0.278 | 10/36 | 0.412 |
| | `generalize` | 0.583 | 0.167 | 6/36 | 0.917 |
| | `introspect` | 0.667 | 0.306 | 11/36 | 0.275 |
| | `eliminate` | 0.556 | 0.250 | 9/36 | 0.564 |

**Not one cell of ten beats the null.** The smallest p in the table is 0.275.
Pooled over all five wordings and both pairs, held-out is **60/360 = 0.167** —
*below* chance — against a pooled anchor of 245/360 = 0.681.

## The wording is not inert, which is what makes this decisive

The easy dismissal of a null elicitation sweep is that the prompts were all bad.
They were not. On the anchor the wording does exactly what it should:

| wording | anchor constant-label cells | anchor row accuracy |
|---|---:|---:|
| `baseline` | 40% | 0.799 |
| `two_groups` | 40% | 0.799 |
| `generalize` | 28% | 0.861 |
| `introspect` | **25%** | **0.875** |
| `eliminate` | 26% | 0.868 |

Telling the model to attend to its own internal state cuts its rate of repeating
one label regardless of the injection from 40% to 25%, and lifts row accuracy from
0.799 to 0.875. The instructions land.

On held-out, the same instructions change nothing:

| wording | held-out constant-label cells | held-out row accuracy |
|---|---:|---:|
| `baseline` | 90% | 0.549 |
| `two_groups` | 76% | 0.604 |
| `generalize` | 85% | 0.576 |
| `introspect` | 76% | 0.590 |
| `eliminate` | 82% | 0.562 |

Between 76% and 90% of the time, under every wording, the model gives the same
answer whichever category was injected. Row accuracy sits at 0.55–0.60, which
reads as chance and means blind. The reader on the identical states is at
0.944–1.000 throughout.

`generalize` is the sharpest case. It says in plain words that the query is a new
member and not a repeat, and instructs against looking for an exact match. It
raised the anchor to 0.861 and left held-out at 0.139.

## The pre-committed reading

The best development wording was `two_groups` at 0.167. **The kill rule required a
wording to clear 0.25 on development, and none did, so the rule fires.** Reported
without further selection, `two_groups` on confirmation is 0.278 with p = 0.412.

`introspect` reaches 0.306 on confirmation and is the highest number in the table.
It was not the development winner and it is not significant (p = 0.275). With 36
twin pairs the standard error at the null is 0.072, so 0.306 is four fifths of one
standard error above chance. Quoting it as a positive would be selecting on the
outcome — the exact error `13` and `15` made, and the reason the split was
declared before the run.

## What this closes

`23`'s negative was not an artifact of one badly chosen prompt. Across five
wordings, including one written specifically to name the missing operation and one
that measurably improved engagement with the task, the model never places an
unseen exemplar into a demonstrated category — while the cheapest possible reader
does it on the same states essentially every time.

**Stop varying the elicitation for this interface.** The live question is whether
training changes it, not whether a sixth prompt would.

This also settles the baseline the handoff's training experiment needs. `21`'s
lesson was that an elicitation-optimized baseline must be frozen before training,
or training takes credit for what wording would have bought. Here wording buys
**nothing** on held-out generalization: the frozen baseline is the floor. Any
held-out ability a trained reporter shows is therefore attributable to training
and not to a better-phrased question.

## Limits

Same as `23`, plus one: this varies instruction framing while holding the readout
fixed at a forced choice between two tokens. It does not test chain of thought,
which needs a generation harness and a different scoring rule. A model allowed to
reason out loud before answering might do better, and that remains untested.
