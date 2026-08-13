# 27 — How much can a free head possibly buy? (pre-run note)

Written before anything ran. Nothing above the closing line will be edited after
seeing results.

## Which project this is

Not the introspection line. This is the **second** SPAR application —
[Deploying Programmatic Attention in Real Transformers](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L)
(Belinda Li) — whose question is:

> Can we run real, readable programs inside a transformer without paying for it
> in speed?

Two experiments already exist for it, and they are recorded in
[`spar-application/PROJECT-BRIEFS.md`](../../spar-application/PROJECT-BRIEFS.md) §2:

- **Routing side, stopped.** A frozen screen over all 64 layer-role-head
  components failed its own gate; influence was spread out rather than running
  down one readable path. That branch does not proceed.
- **Speed side, an informative collapse.** One released GPT-2 attention program
  was rewritten by hand into an exactly equivalent form that never builds the
  big square attention matrix. On its own it is **18.6× faster** at a
  1024-token input. Put back inside a real GPT-2 attention block it is
  **1.089× faster**, against a 1.25× bar frozen before the run. The brief's own
  reading of that: *"the algebra was never the bottleneck — partial-head
  projection and dispatch are."*

That reading has never been tested. It is a guess about why the number was small.

## Mode: frontier, and the cheapest possible capacity check

The standing rule is to run the capacity check first — the version of the task
with the answer in plain sight, nothing else changed — before spending anything
on mechanism. Here that check is almost embarrassingly simple, and nobody has
run it:

> **Delete the head entirely and time the model.**

A head that has been deleted costs nothing at all. No program can ever be
cheaper than that. So the speed of a model with *k* heads deleted is a hard
ceiling on the speed of the same model with *k* heads replaced by any program,
however clever, however well written in CUDA. If the ceiling is below the bar,
no amount of kernel engineering gets there and the whole speed workstream needs
rescoping at this model size.

Deleting heads wrecks what the model says. That does not matter here: this arm
is a **stopwatch**, not a model anybody would use. It measures how much of the
running time a head is responsible for, and nothing else.

## What I am about to do

GPT-2 small (124 million parameters), real released weights, evaluation mode,
whole sequence at once, no cached decoding — the same interface the earlier
benchmark supports.

Three arms, timed at every combination of input length and batch size:

| arm | what it is | what it tells us |
|---|---|---|
| **stock** | the unmodified model | the denominator |
| **deleted `k`** | `k` heads per layer removed outright with the library's own head-pruning call | the ceiling. No program beats this |
| **programmatic `k`** | `k` heads per layer replaced by the exact rewritten form from the earlier study — it still computes each head's value vector and still projects the result back out, but never builds the square matrix | what is actually reachable today |

`k` runs 0, 1, 2, 3, 4, 6, 8, 10 out of 12 — so coverage from none to five-sixths
of every layer. The paper this project builds on replaces **25% of heads**, which
is `k = 3`; that column is the one the project's own literature cares about.

Input lengths 64, 128, 256, 512, 1024 and batch sizes 1 and 8, matching the
frozen grid the earlier benchmark used so the new numbers sit next to the old
ones without an apples-to-oranges problem.

Two machines' worth of behaviour, not one: **processor** (single-threaded, the
setting the old protocol froze) and **graphics chip** (Apple's MPS backend). They
fail differently — on the processor the cost is arithmetic, on the graphics chip
it is the fixed price of launching each operation — and a result that only holds
on one of them is not a deployment result.

Timing is paired and alternated, medians over 30 blocks, with the same
uncertainty method the earlier benchmark used.

## The three numbers this produces

1. **The ceiling.** How much faster is GPT-2 with `k` heads deleted, at each
   length. Directly answers "is 1.25× even on the table".
2. **The gap.** Ceiling minus what the real rewritten program achieves. This
   *is* the brief's suspected bottleneck, measured rather than asserted — it is
   the price of keeping the value and output projections plus the cost of
   dispatching extra operations.
3. **The break-even coverage.** For each length, the smallest fraction of heads
   that would have to be programmatic to reach 1.25× end-to-end. If that number
   is above 100%, the bar is unreachable at this size, full stop.

## What each outcome would mean — including the dull one

- **Ceiling below 1.25× at 25% coverage for every length tested.** Then the bar
  the earlier study missed was never reachable with one head, or three, and the
  1.089× was not a failure of the implementation. The honest conclusion is that
  at this model size and these lengths, programmatic attention is an
  *interpretability* proposition that costs a little speed, not a speed
  proposition. That reframes the workstream and is worth saying.
- **Ceiling well above the bar but the real program far below it.** Then the
  brief's guess is right, the loss is in projection and dispatch, and there is a
  concrete kernel target with a number attached. That is the positive version
  and it points straight at the project's own CUDA workstream.
- **Both above the bar.** Then the 1.089× was an artifact of replacing exactly
  one head, multi-head replacement pays, and the next step is obvious.

All three change what happens next, which is the test of whether this is worth
running.

## What it costs

One 550 MB model download, then minutes. No training, no GPU memory pressure
worth worrying about — GPT-2 small is roughly a quarter of the smallest model
this repository normally loads. Under half an hour of wall clock.

## Predictions, recorded before the run

Written down so they can be wrong in public.

1. At 1024 tokens, batch 1, processor: deleting 3 of 12 heads buys **less than
   1.15×** end-to-end. My centre guess is 1.06–1.10×.
2. Deleting **10 of 12** heads at 1024 tokens still buys **less than 1.6×**.
3. The break-even coverage for 1.25× at 1024 tokens is **above 50%** of all
   heads on the processor.
4. The real programmatic arm lands within 0.05× of the deletion ceiling at
   k = 3 on the processor — i.e. **the brief's "projection and dispatch"
   explanation is wrong**, and the small speedup is simply because one head is a
   small share of the work.
5. On the graphics chip at short lengths, the programmatic arm is **slower than
   stock** — extra separate operations cost more than the arithmetic they save.

Prediction 4 is the one I most expect to lose, and it is the one that matters:
it decides whether the next step is a kernel or a rescope.

---

*Everything below this line was written after the run.*

## What happened

Both gates passed. 3,150 timing blocks, 210 cells, no arm produced a bad number,
and the deleted-head arm bounded the programmatic arm everywhere — which is the
check that the stopwatch itself is sound, since a head that is gone cannot cost
more than a head that was replaced.

**In one sentence: I was wrong about the size of the prize and right about where
it is lost — the ceiling is far higher than I predicted, and the exact lowering
is already sitting on it.**

### Four of five predictions resolved, and I lost the three that were about size

| # | prediction | actual | |
|---|---|---|---|
| 1 | 3 of 12 heads buys < 1.15× at 1024 tokens | **1.187×** | wrong |
| 2 | 10 of 12 heads buys < 1.6× | **2.091×** | wrong |
| 3 | break-even coverage above 50% | **34%** | wrong |
| 4 | the program lands within 0.05× of the ceiling | **0.015× apart** | right |
| 5 | the program is slower than stock at short lengths | **0.980× at 64 tokens** | right |

Wrong three times in the same direction is a pattern, not bad luck. I estimated
the prize by counting multiplications, and multiplications are not what the
explicit attention implementation spends its time on. Removing one head from
every layer should have been worth about 2% by arithmetic; it is worth 5.6%.
The extra comes from the memory traffic of building and reading a square
token-by-token score matrix, which counting operations does not see.

### The ceiling, and how much of it the real program gets

End-to-end speedup at 1024 tokens. "Deleted" is the ceiling — the head is gone.
"Program" is the exact lowering, fused across all replaced heads in a layer.

| heads replaced (of 12) | processor deleted | processor program | graphics deleted | graphics program |
|---|---:|---:|---:|---:|
| 1 | 1.056 | 1.048 | 1.038 | 1.025 |
| 3 *(25%, the literature's coverage)* | 1.187 | **1.172** | 1.131 | **1.099** |
| 4 | 1.237 | 1.243 | 1.175 | 1.142 |
| 6 | 1.444 | 1.423 | 1.303 | 1.231 |
| 8 | 1.724 | 1.669 | 1.444 | 1.350 |
| 10 | 2.091 | 2.005 | 1.631 | 1.468 |

The program captures **96–100% of the ceiling on the processor** and **90–99% on
the graphics chip**. There is essentially nothing between running the program and
deleting the head outright.

### The thing this was run to decide

The project brief reads the earlier 1.089× as *"the algebra was never the
bottleneck — partial-head projection and dispatch are"*, and proposes measuring
where that overhead goes and whether a fused projection recovers any of the
isolated 18.6×. **That diagnosis is wrong, and this measurement is the fused
version it proposed.** Fusing the replaced heads into one projection in and one
out per layer gets within 1.5 percentage points of a head that does not exist.
There is no overhead left to recover, because there was almost none to begin with.

The earlier 1.089× was small for a much duller reason: **one head out of twelve is
a small share of the work.** Replace three and you get 1.19×; replace ten and you
get 2.09×. The lever is **coverage**, not kernel engineering.

That said, dispatch is not exactly zero, and the run says where it lives: the
program keeps 96% of the ceiling on the processor but only 90% on the graphics
chip at high coverage, and at 64 tokens it is **slower than doing nothing** (0.980×
on the graphics chip, 0.992× on the processor). Fixed per-operation cost is real,
it is a graphics-chip and short-sequence phenomenon, and it is worth roughly a
tenth of the prize rather than all of it.

### How much coverage the bar actually needs

Smallest fraction of heads that must be programmatic to reach the 1.25× the
earlier study missed:

| | 64 tok | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| processor, deleted | never | never | 67% | 42% | **34%** |
| processor, program | never | never | 83% | 51% | **34%** |
| graphics, program (batch 1) | never | never | never | never | 53% |
| graphics, program (batch 8) | never | never | never | 62% | 45% |

Two things fall out of that table.

**The bar is reachable — but not where the literature is standing.** At 25%
coverage, the best number anywhere in this run is 1.187×, and every other
configuration is between 1.10× and 1.14×. Nothing at 25% reaches 1.25× on any
device, batch or length tested.

**Short context kills it outright.** At 64 and 128 tokens the bar is unreachable
at *any* coverage, on either device, even with the heads deleted entirely.

### The trade nobody has put in one place

This study measures cost only. Both arms deliberately break the model. But the
cost number can be set against the published quality number, and the result is
not flattering. Hayes, Li and Andreas report that replacing 25% of heads with
programmatic surrogates costs **about 16% higher perplexity on average across
three models** *(from the paper's abstract — I have not read the full paper, and
that figure is an average across models, not GPT-2 specifically)*.

So at the coverage the field actually proposes, the trade at this model size is
roughly:

> **give up 16% of your language modelling, gain 10–19% of your speed.**

That is a bad trade, and it is bad for a reason this study can name: at 25%
coverage the speed prize is capped at 1.19× *even if the replaced heads were free*.
The interpretability argument for programmatic attention does not depend on this
and is untouched. The efficiency argument, at GPT-2 scale and 1024 tokens, does
not currently clear its own bar.

## What is not established

The deleted-head arm is a stopwatch and its output is meaningless; so is the
programmatic arm's, since it imposes one released head's program on heads that
never had it. Nothing here is a claim about quality — the 16% above is somebody
else's measurement, quoted from an abstract. Nothing here covers cached
decoding, training, CUDA, multi-threaded processors, other model sizes, other
programs, or lengths past 1024. Three of those limits are the subject of
[`28`](28-three-ways-the-ceiling-could-be-wrong.md), which was written and frozen
while this run was still going.

## What I would do next

1. **[`28`](28-three-ways-the-ceiling-could-be-wrong.md)** — already running. The
   denominator here contains GPT-2's vocabulary projection at every position,
   which nobody pays in deployment, and the baseline is the slow attention nobody
   deploys. Its smoke already shows the second one bites: at 2048 tokens and 25%
   coverage, the ceiling is 1.24× against explicit attention but **1.12× against
   the fast implementation**.
2. **Coverage is now the question, so measure how much coverage is even
   available.** The released programs are one file. For each, materialize its
   attention matrix at growing lengths and ask a mechanical question: does the
   number of non-zero entries per row, or the matrix rank, stay bounded as the
   sequence grows? Either one means an exact linear-time form exists. That
   converts "the lever is coverage" into a number, and needs no GPU.
3. **Do not** build a fused CUDA kernel for this. That was the brief's proposed
   next step and this run says the ground it would recover is 1–4 percentage
   points on the processor and about 10 on the graphics chip.
