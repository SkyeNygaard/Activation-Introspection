# Does looking inside a model help you find what a fine-tune broke?

A model fine-tuned on one narrow bad habit starts behaving badly on unrelated
things. An auditor handed the before-and-after pair has far too many questions to
test them all, and must guess which ones to spend the budget on.

**Short answer: looking inside beats reading the outputs two-to-one, and both are
beaten by a list of which questions broke under a previous fine-tune. Looking inside
adds nothing on top of that list.**

![result](results/ladder.png)

## The ladder, by what the auditor already has

Three published fine-tunes of `Llama-3.2-1B-Instruct` — bad medical advice, risky
financial advice, reckless sports advice — tested on 300 everyday questions about
animals, food, tools and vehicles that have nothing to do with any of them. 7,200
answers, each scored by a judge blind to which version wrote it.

Target: where the fine-tune *changed* things, not where the model was always
mediocre. Zero is guessing.

| what the auditor has | how they rank the questions | score |
|---|---|---:|
| two copies of the model | compare their outputs | +0.13 |
| two copies, and judged labels for some topics | compare their internals | +0.28 |
| **one previous fine-tune, already judged** | **a list of what broke then** | **+0.55** |
| both of the above | the list, plus the internals | +0.54 |

## Why the internal signal loses

Damage on a question splits in two: a part shared with the sibling fine-tunes (43%)
and a part specific to this one (57%).

|  | shared with siblings | specific to this fine-tune |
|---|---:|---:|
| internals, fitted reader | +0.30 to +0.38 | +0.02 to +0.11 |
| outputs | +0.14 to +0.18 | −0.03 to +0.06 |

Every interval on the right includes zero. **Nothing tested predicts what this
particular fine-tune broke.** The internal signal correlates with the question list
at +0.30 to +0.38 — it is an imperfect proxy for how fragile a question is, which
the list measures directly and better. It beat the outputs because it is a better
proxy, not because it sees the fine-tune in front of it.

## What survives

- **Internals beat outputs**, +0.28 against +0.13, on all three fine-tunes,
  surviving controls for question length, answer shortening, floor effects and a
  verified-exact adapter toggle. That comparison stands; it is now a comparison
  between two methods that a third beats.
- **Direction, not size.** The plain magnitude of the internal change is the best
  single number on two fine-tunes and worth nothing on the third. Only the direction
  replicates.
- **The advantage lives before the readout** — peak at depth 13 of 16, worse at the
  final layer, so it is not simply a restatement of the output.
- **The internal reader transfers**: fitted on one fine-tune, applied to another, it
  scores +0.50 to +0.53 against +0.49 to +0.61 on its own. The three fitted
  directions overlap at +0.60 to +0.64, where unrelated directions would overlap at
  0.02. There is one shared axis.
- **One cheap look inside is worth a whole generation outside** — replaying entire
  answers through both models gets the outputs to +0.24, level with what one forward
  pass inside already gave.

## Reading order

1. [notes/00-plan.md](notes/00-plan.md) — the plan, written before anything ran, with
   one amendment made before the data existed.
2. [notes/01-capacity-check.md](notes/01-capacity-check.md) — why the model is 1B.
3. [notes/03-three-fine-tunes.md](notes/03-three-fine-tunes.md) — internals beat
   outputs. **Its headline is superseded by 05.**
4. [notes/04-plan-attack-the-result.md](notes/04-plan-attack-the-result.md) — four
   free checks designed to break it, written before running them.
5. [notes/05-the-cheap-baseline-wins.md](notes/05-the-cheap-baseline-wins.md) —
   **the result.** Including a leak found and fixed on the way.

## Reproducing

Models and adapters are published by others; nothing here was trained.

```bash
./scripts/run_all.sh && ./scripts/run_forced.sh
python scripts/judge.py --tag llama1b_bad-medical-advice
python scripts/analyze.py --tag llama1b_bad-medical-advice
python scripts/pooled.py && python scripts/attack.py && python scripts/attack2.py
python scripts/attack3.py && python scripts/specific.py && python scripts/plot.py
```

Uses the `activation-introspection` virtual environment next door; no dependency was
added to it, so its frozen protocol hashes are untouched.

| script | what it does |
|---|---|
| `collect.py` | signals from the question alone, plus the answers that give ground truth |
| `judge.py` | scores every answer, blind to which version wrote it |
| `analyze.py` / `pooled.py` | the four-way comparison, one fine-tune and all three |
| `checks.py` | depth profile, paired tests, length and shortening controls |
| `forced.py` | the fair-at-higher-cost control |
| `attack.py` / `attack2.py` / `attack3.py` | the cheap baselines that overturned it |
| `specific.py` | what predicts the fine-tune-specific part of the damage |

## Limits

One base model, one size, one kind of fine-tune (low-rank adapters from a single
group). Qwen2.5-7B loads on this machine and then runs out of memory on its first
forward pass, so scale is untested. Whether the fragile-question list transfers
across model families is the open question that decides how useful it is, and is
being tested.
