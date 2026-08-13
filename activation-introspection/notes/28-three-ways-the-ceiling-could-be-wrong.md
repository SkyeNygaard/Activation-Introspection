# 28 — Three ways the ceiling could be wrong (pre-run note)

Written **while [`27`](27-how-much-can-a-free-head-possibly-buy.md) was still
running**, and deliberately so. See the disclosure below for exactly how much of
`27` I had seen.

## Disclosure, first

`27` was running when this was written. Four lines of its progress log had
scrolled past: processor, deleted-head arm, one head removed, at the four
shortest lengths, and two head removed at three lengths. Those numbers were
between 1.02× and 1.07×. **No batch-8 cell, no graphics-chip cell, no
programmatic-arm cell, and nothing at all above k = 2 had been produced.**

I am writing this now rather than after `27` finishes because the three
objections below came out of re-reading my own design, not out of its results,
and that is a much better reason to trust them. If I wait, nobody can tell the
difference.

## The problem

`27` measures how much faster GPT-2 gets when heads are deleted. To turn that
into a number, it has to divide by *something* — and it divides by the running
time of the whole `GPT2LMHeadModel`, at every position, with attention computed
the slow explicit way, on sequences of at most 1024 tokens.

Every one of those four choices is defensible. Three of them are also, on
reflection, **choices that push the answer in a known direction**, and I picked
them without noticing.

### Threat 1 — the vocabulary projection is in the denominator

The last thing GPT-2 does is multiply by a 768 × 50,257 matrix to turn each
position into word scores. At 1024 tokens that single step is roughly **a
quarter of all the arithmetic in the model** — comparable to several whole
transformer layers.

`27` computes it at every position. Anyone actually running a model does not:
when you feed in a prompt, you only need word scores for the *last* position.
So `27`'s denominator contains a large cost that a real deployment does not pay,
and dividing by a too-large denominator makes every speedup look **smaller than
it is**.

Rough size of the error, from arithmetic alone: at 1024 tokens and ten of twelve
heads removed, including that projection gives about 1.24× where excluding it
gives about 1.36×. Those two numbers **fall on opposite sides of the 1.25× bar**.
This is not a rounding detail; it can flip the verdict.

### Threat 2 — the baseline is the slow attention

`27` uses the explicit implementation, which builds the full square
token-by-token score matrix in memory. Nobody deploys that. The standard
implementation (`sdpa`, the family that includes FlashAttention) never builds the
matrix at all — which is *the same trick* the programmatic lowering is selling.

So `27` may be measuring a program beating a straw man. Against the baseline
people actually run, the advantage should shrink, possibly a lot. This one pushes
the answer the other way: `27` is too **generous**.

### Threat 3 — 1024 tokens is the short end

Attention cost grows with the square of the length; everything else grows
linearly. So the share of time attention is responsible for — and therefore the
whole prize — **grows without limit as context gets longer**. GPT-2's position
table stops at 1024, which is short by any modern standard.

A negative result at 1024 tokens would say nothing about 8k or 128k. This is the
difference between "programmatic attention does not pay" and "programmatic
attention does not pay yet, and here is the length where it starts to".

## What I am about to do

The same measurement as `27`, with those three choices flipped, one axis at a
time so the effect of each is separable:

| axis | `27` | `28` |
|---|---|---|
| what is timed | whole model including word scores at every position | the transformer stack only, no vocabulary projection |
| attention implementation | explicit (`eager`) | both explicit and the standard fast one (`sdpa`) |
| length | up to 1024 | up to 4096 |

Going past 1024 needs one honest hack, stated plainly: GPT-2's table of position
embeddings only has 1024 entries, so it is **replaced with a larger randomly
filled one**. The model's output becomes meaningless. That is fine and it is the
same licence `27` already takes — the deleted-head arm has meaningless output
too. Both arms are stopwatches. Nothing in either note is a claim about what the
model *says*, only about what it *costs*. A run at 4096 tokens is "GPT-2's
arithmetic at 4096 tokens", not "a working 4k model".

To keep this affordable it runs on the graphics chip only, batch 1, at
k = 3, 6, 10, with 9 timing blocks instead of 15. The 4096-token explicit arm
holds a 12 × 4096 × 4096 score matrix — about 800 MB — which fits, where 8192
would need 3.2 GB and is not attempted.

## What each outcome would mean

- **Threat 1 is large and the answer flips.** Then `27`'s headline understates
  the case, the right denominator excludes the vocabulary projection, and the
  quotable number is the one from here.
- **Threat 2 is large.** Then the honest claim shrinks to "faster than an
  implementation nobody deploys", which would be the single most important thing
  either note produces, and the one most likely to be missing from work in this
  area generally.
- **Threat 3 shows a crossing.** Then there is a context length at which this
  starts to pay, it can be named, and a negative at 1024 stops being a verdict on
  the idea.
- **All three small.** Then `27` stands as written, which is the dull outcome and
  still worth the twenty minutes, because right now I cannot say that.

## What it costs

No new downloads. Under twenty minutes on the graphics chip. Nothing runs on the
processor while `27` is still using it single-threaded.

## Predictions, recorded before the run

1. Threat 1 is the biggest of the three, worth **8 to 15 percentage points** of
   speedup at the high-coverage end.
2. Threat 2 costs the deleted-head arm **less than a third** of its advantage —
   the fast implementation saves memory traffic, but the projections it cannot
   avoid are most of what a deleted head takes with it.
3. There is no crossing below 4096 tokens: even with the vocabulary projection
   removed, the fast implementation in place, and 4096 tokens, **25% coverage
   stays under 1.25×**.
4. Prediction 3 fails at high coverage: 10 of 12 heads at 4096 tokens with no
   vocabulary projection **does** clear 1.25×.

---

*Everything below this line was written after the run.*

## What happened, including the part that went wrong

**In one sentence: two of the three objections were real and large, and the one
that matters most is the second — measured against the attention implementation
people actually deploy, roughly half the advantage disappears.**

### First, the run that failed its own gate

The full grid was run once and **failed the `ceiling_bounds_program` gate.** One
cell of forty-eight came out impossible: at 1024 tokens with 3 heads replaced,
explicit attention, the *programmatic* arm measured 1.135× against a
*deleted-head* arm of 1.088×. A head that has been removed cannot cost more than
a head that has been replaced, so that ordering is a statement about the
stopwatch, not about the model.

It also disagreed with `27`, which had measured 1.131× for the same arm, device
and length *with* the vocabulary projection still in the denominator. Removing a
large fixed cost can only raise a speedup. Both facts pointed the same way.

My first explanation was wrong, and I am leaving it here because the way it died
is the useful part. I reasoned that reaching 4096 tokens requires enlarging
GPT-2's precomputed triangular mask buffer to 4096 × 4096, that the explicit
attention path slices that buffer on every layer, and that this cost was present
in every cell including the short ones. It is a tidy story and it fits the
evidence I had.

Then the same grid was run again, unchanged, **with the same enlarged buffer**,
and the bad cell came back at 1.165× — matching the capped run's 1.164× and
`27`'s ordering. The buffer was never the problem.

**The problem was me.** While that first grid was running I was doing other work
on the same machine: reading its summary, extracting tables, running the
formatter. The damage is confined to the *first arm measured*, at the *two
shortest lengths* — the cells with the smallest absolute differences and the
most iterations per block, so the most sensitive to a burst of competing work.
Every later cell in that run reproduces to within about 1%.

| cell | contaminated run | clean rerun |
|---|---:|---:|
| explicit, deleted, 3 heads, 512 | 1.076 | 1.119 |
| explicit, deleted, 3 heads, 1024 | 1.088 | 1.165 |
| explicit, deleted, 3 heads, 4096 | 1.261 | 1.264 |
| fast, deleted, 10 heads, 4096 | 1.951 | 1.946 |

Two lessons, both cheap to state and both learned the expensive way. **A paired
alternating design protects against slow drift, not against a burst.** And I had
written in `27`'s own procedure that nothing else may run during timing, then
broke it in the next study while writing that study up.

The clean rerun **passes both gates** and is the source of every number below.
The failed run's artifacts are kept rather than deleted, with the gate result
recorded as false.

### Threat 1 — the vocabulary projection. Real, and bigger than I guessed.

Comparing like with like: same device, same batch, same attention, same lengths,
the only difference being whether GPT-2's word-score projection is inside the
timed region.

| heads replaced | with the projection (`27`) | without it | difference |
|---|---:|---:|---:|
| 3, deleted | 1.131 | 1.164 | +3.3 points |
| 10, deleted | 1.631 | 1.890 | **+25.9 points** |
| 10, programmatic | 1.468 | 1.655 | +18.7 points |

I predicted 8 to 15 points at the high-coverage end. It is **26**. So `27`
understates the case, and it understates it most exactly where the numbers are
most interesting. Prediction 1 was wrong in size, right in direction and right
that this was worth checking.

### Threat 2 — the baseline. The most important result in either note.

At 1024 tokens, no vocabulary projection, changing only which attention
implementation the *unmodified* model uses:

| heads replaced | vs explicit attention | vs the fast one | advantage lost |
|---|---:|---:|---:|
| 3, deleted | 1.165 | 1.107 | 36% |
| 10, deleted | 1.870 | 1.468 | **46%** |
| 10, programmatic | 1.655 | 1.260 | **60%** |

And at 4096 tokens, 10 heads deleted: 3.04× against explicit attention,
**1.95× against the fast one** — 54% of the advantage gone.

I predicted this would cost less than a third of the advantage. It costs
**roughly half**. Prediction 2 is wrong, and wrong in the direction that hurts
the idea.

This is the finding I would put first. The pitch for programmatic attention is
that it avoids building the big square score matrix — but the standard fast
implementation **already does that**, and has for years. Measuring against the
explicit implementation, as the earlier benchmark and this project's first study
both did, compares a new idea to something nobody runs. About half of the
apparent prize is not the program's doing at all.

### Threat 3 — context length. Real, large, and the idea's best argument.

Deleted-head ceiling against the fast implementation, no vocabulary projection:

| heads replaced | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| 3 | 1.077 | 1.107 | 1.129 | 1.166 |
| 6 | 1.184 | 1.230 | 1.311 | 1.422 |
| 10 | 1.366 | 1.468 | 1.630 | 1.946 |

Programmatic, same conditions, 10 heads: 1.176 → 1.260 → 1.423 → **1.723**.

The prize grows steadily with length and there is no sign of it flattening. This
is the honest case for the idea: it is a **long-context** proposition. Nothing
here reaches modern context lengths — 4096 is where GPT-2's arithmetic was still
affordable on this machine — and the trend says the interesting regime is past
where I can measure.

### The two predictions I got right

Prediction 3: at 25% coverage, against the fast implementation, with the
vocabulary projection removed and 4096 tokens, the programmatic arm stays under
1.25×. It reaches **1.132×**. Correct.

Prediction 4: 10 of 12 heads at 4096 tokens under those conditions does clear
1.25×. It reaches **1.723×**. Correct.

## Where this leaves the project's question

Putting `27` and `28` together, under the least favourable-to-me and most
realistic conditions available here — fast attention, no vocabulary projection —
GPT-2 small at 1024 tokens:

| coverage | programmatic speedup | clears 1.25×? |
|---|---:|---|
| 25% *(the literature's setting)* | 1.06× | no |
| 50% | 1.13× | no |
| 83% | 1.26× | just |

At 4096 tokens the same three become 1.13×, 1.34×, 1.73×.

Set against the ~16% average perplexity cost the source paper reports for 25%
coverage, the trade at 25% is **1.06× faster for 16% worse language modelling**
at 1024 tokens, or 1.13× at 4096. That is not a deployment case. The case only
starts to work at coverage far beyond what the quality results support, or at
context lengths beyond what this machine can reach.

None of that touches the interpretability argument, which never depended on
speed.

## What I would do next

1. **Stop measuring against explicit attention.** Any future number in this
   project should be quoted against the fast implementation. This applies
   retroactively to the 18.63× isolated-operator figure, which compared against a
   cached dense matrix multiply — the most generous baseline available.
2. **Go long, not wide.** The length trend is the only axis where the idea is
   winning, and it had not flattened at 4096. A model with a real long context
   would say whether this becomes a genuine deployment case at 32k.
3. **Then coverage.** How many released programs even admit an exact
   linear-time form is now the binding constraint on everything above, and it
   costs no GPU: materialize each released program's matrix at growing lengths
   and check whether non-zeros per row, or rank, stay bounded.
