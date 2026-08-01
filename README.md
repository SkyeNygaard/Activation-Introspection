# activation-introspection

Can a language model report on interventions made directly to its own
activations — using access an outside observer would not have?

This repo is a laptop-scale testbed for that question, built around the confound
that decides it: a model that describes an injected concept may be reading its
internal state (**introspection**), or may simply be noticing that its own output
has drifted and inferring backwards (**behavioural inference**). Almost every
positive result in this area is compatible with the second explanation.

Every experiment here runs an **observer arm**: a clean copy of the model sees
only the intervened model's *output* and answers the same question. The headline
metric is the gap between the two, not the introspecting model's raw score.

See [`notes/01-problem-space.md`](notes/01-problem-space.md) for the full argument
and [`notes/02-experiment-plan.md`](notes/02-experiment-plan.md) for the
pre-registered conditions.

## Status

Full pipeline implemented and running: concept banks, interventions, both
elicitation modes, the observer arm, bootstrap intervals, and a scale ladder.
Sweep results are being collected; treat any number currently in `results/` as
preliminary until it appears in [`notes/03-lab-notebook.md`](notes/03-lab-notebook.md).

## The design

Six arms per cell, all run in one function so they cannot acquire separate
prompts, seeds, or bugs:

| arm | rules out |
|---|---|
| `clean` | response bias — the model saying YES regardless |
| `concept` | — |
| `shuffled` | "any perturbation triggers a report" (matched norm, permuted coords) |
| `random` | same, looser null |
| observer on `concept` | **behavioural inference** — a clean model reading only the output |
| observer on `shuffled` | observer-side bias |

The observer is the *same weights* with a fresh context, which holds capability
fixed and varies only whether the answer can draw on internal state. A stronger
observer would confound "no privileged access" with "better at the task".

Two gates decide whether a cell is interpretable at all, and `analysis.py` marks
a cell INVALID if either fails: the shuffled control must not detect above
chance, and the behavioural effect must be non-zero.

**The pre-verbalization arm is the detection score.** It is read at the first
generated position of a fresh context, so the model has emitted no task text and
behavioural inference has nothing to work from. There is deliberately no observer
comparison there — an observer would have nothing to read, which is the point.

## Setup

```bash
make setup
```

Run the pipeline:

```bash
uv run python scripts/run_sweep.py --model qwen-1.5b --seeds 5
```

Scale ladder (loads and frees one model at a time — holding two resident is what
pushes a 24 GB machine into swap):

```bash
uv run python scripts/run_ladder.py --models qwen-0.5b,qwen-1.5b,qwen-3b
```

Re-analyse a saved sweep without touching a model:

```bash
uv run python scripts/analyze.py results/ladder.jsonl
```

Python 3.12 via `uv`. Weights download into `./hf_cache` on first run so the
project stays self-contained; `make clean-cache` removes them.

## Smoke test

```bash
make smoke
```

Proves four things before any experiment is worth running: the model loads and
generates on this machine, hooks fire on the right blocks and are removed
afterwards, concept directions are mutually distinguishable, and injection at a
mid layer measurably changes the output while a matched-norm random direction
changes it less specifically.

Override the defaults:

```bash
uv run python scripts/smoke_injection.py --model qwen-7b --concept volcano --layer 18 --strength 3.0
```

## What's here

| Path | Purpose |
|---|---|
| `src/introspect/models.py` | Loading, device/dtype selection, architecture-agnostic block access |
| `src/introspect/hooks.py` | `capture` and `intervene` context managers; `Intervention` spec |
| `src/introspect/concepts.py` | Contrast-pair concept vectors, plus matched-norm random and shuffled controls |
| `src/introspect/prompts.py` | Detection, identification, and observer-arm elicitation |
| `scripts/smoke_injection.py` | End-to-end plumbing check |
| `tests/` | Position-mask and hook-hygiene tests against a stub model — no weights needed |

## Design notes

**Strength is normalised to the measured residual norm** at the injection layer
(`Intervention(normalize=True)`). Residual norm grows several-fold with depth, so
a fixed alpha across a layer sweep yields a monotone curve that is purely an
artefact of scale.

**Position masking is load-bearing.** `positions="generated"` is what makes the
pre-verbalization condition meaningful: if the model is asked to detect an
injection before emitting any task text, behavioural inference has nothing to
work from. The mask logic is unit-tested for exactly this reason.

**Controls are matched-norm, not absent.** `shuffled_control` permutes the
concept vector's coordinates, preserving norm and coordinate distribution while
destroying direction — a tighter null than Gaussian noise.

## Scale caveat

Introspective report is plausibly emergent. A 1.5B model may fail every condition
without that saying much about frontier models. The intended claim is about the
measurement apparatus and where the effect first appears across the scales that
fit on a laptop — not a frontier-model result.

## Licence

MIT.
