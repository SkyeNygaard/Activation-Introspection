# Does looking inside a model help you find what a fine-tune broke?

A model fine-tuned on one narrow bad habit starts behaving badly on unrelated
things. An auditor handed the before-and-after pair has far too many questions to
test them all, and has to guess which ones to spend the budget on.

**They can guess from the outside** — feed a question to both copies, compare the
next-word probabilities. **Or from the inside** — compare the numbers the two
copies compute along the way. This measures whether the inside is worth the
trouble, with the outside method given a fair fight rather than a handicap.

![result](results/ladder.png)

## The answer

Across three published fine-tunes of `Llama-3.2-1B-Instruct` — bad medical advice,
risky financial advice, reckless sports advice — tested on 300 everyday questions
about animals, food, tools and vehicles that have nothing to do with any of them:

| ranking the auditor uses | how well it matches the damage |
|---|---:|
| outputs, one number | +0.13 |
| outputs, a fitted reader on twelve output-side numbers | +0.11 |
| internals, one number | +0.20 *(worth nothing on one of the three)* |
| **internals, a fitted reader on the internal difference** | **+0.26** |

The fitted internal reader beats the outputs by **+0.14 [+0.06, +0.22]**. Judging
noise caps any signal at about **+0.48**, so that is over half of what is
attainable against roughly a quarter.

Three things sharpen it:

- **On the questions where the outputs barely moved** — named in advance as the
  place internals could win — the outputs fall to +0.06 and the internals hold at
  +0.30. The margin roughly doubles.
- **The advantage lives before the output.** Measured at every depth, the internal
  signal peaks at depth 13 of 16 and gets *worse* at the final layer, where the
  output is read from.
- **One cheap look inside is worth a whole generation outside.** Replaying entire
  answers through both models and accumulating the disagreement — which costs the
  generation the ranking was meant to avoid — gets the outputs to +0.24, level with
  what one forward pass inside already gave (+0.26, difference not distinguishable).
  At that higher cost internals still win, +0.35 against +0.24, in all three
  fine-tunes separately.

**What does not replicate:** the plain *size* of the internal change wins on two
fine-tunes and is worth nothing on the third. What survives is reading the
*direction* of the change. "How much did it move" is the wrong question.

## Reading order

1. [notes/00-plan.md](notes/00-plan.md) — what was going to be done and why, written
   before anything ran, with one amendment made before the data existed.
2. [notes/01-capacity-check.md](notes/01-capacity-check.md) — why the model is 1B
   and not 0.5B or 7B.
3. [notes/02-medical-result.md](notes/02-medical-result.md) — the first fine-tune,
   before replication changed the headline.
4. [notes/03-three-fine-tunes.md](notes/03-three-fine-tunes.md) — **the result.**

## Reproducing

Models and adapters are published by others; nothing here was trained.

```bash
./scripts/run_all.sh                                   # ~40 min per fine-tune
python scripts/judge.py --tag llama1b_bad-medical-advice
python scripts/analyze.py --tag llama1b_bad-medical-advice
python scripts/pooled.py && python scripts/plot.py
./scripts/run_forced.sh && python scripts/pooled_forced.py
```

Uses the `activation-introspection` virtual environment next door; no dependency
was added to it, so its frozen protocol hashes are untouched.

| script | what it does |
|---|---|
| `collect.py` | signals from the question alone, plus the answers that give ground truth |
| `judge.py` | scores every answer, blind to which version wrote it |
| `analyze.py` | the four-way comparison for one fine-tune |
| `checks.py` | depth profile, paired tests, length and shortening controls |
| `pooled.py` | all three fine-tunes, resampling questions not rows |
| `forced.py` / `pooled_forced.py` | the fair-at-higher-cost control |

## Limits

One model, one size, one kind of fine-tune (low-rank adapters from a single
group). 7B loads on this machine and then runs out of memory on its first forward
pass, so scale is untested — and it is the first thing worth doing next. The
measurement says *where* behaviour changed, not what changed or why.
