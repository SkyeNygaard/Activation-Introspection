# Output-ready state transfer: a located site that did not confirm out of sample

Three runs in one day, two frozen protocols, and no reporting row. The blind
anchors missed the site by one block; the site was then located on development
data and failed its held-out confirmation. Both stops are recorded here.

Run date: **2026-08-11**

## Question

[`09`](09-natural-state-pilot.md) stopped at its reachability gate: replacing the
layer-9 route-marker residual did not make `Qwen2.5-3B-Instruct`'s ordinary
two-hop answer follow the donor state, so no reporting row ran. That note closed
the route site and named the reopen condition — localize a token and layer whose
cross-patch does change the ordinary answer, on a reliable task and a disjoint
development bank, before exposing a fresh report bank.

This pilot takes the easiest version of that condition. It transplants an
**output-ready** state instead of a hidden intermediate: the residual at the last
pre-answer token of an arithmetic problem the model solves itself. That is the
position whose residual feeds the answer, so if any single natural state is
causally load-bearing for the next token, it is this one. The claim a positive
would support is correspondingly narrower than the route design's, and was
labelled as such before the run.

## Design

| element | choice |
|---|---|
| donor task | five twin pairs of single-digit arithmetic, one operand apart |
| hidden class | parity of the answer: `+1` even, `-1` odd |
| capture site | last pre-answer token of the clean prompt |
| anchor layers | 9, 21, 26, named before the run; earliest passing selected |
| development bank | `4+4\|5`, `3+3\|4`, `2+2\|3`, `1+1\|2`, `9-9\|8` |
| held-out bank | `6+2\|3`, `5+1\|2`, `7-3\|2`, `9-7\|6`, `4-4\|3` |

Each bank's ten answers are the ten digits, so parity is the only property that
separates the classes across pairs — within a pair the even twin is always the
smaller answer, but the reporter never contrasts within a pair.

Both banks were frozen together. The development bank selects the layer and is
never reported on; the held-out bank supplies the reporter's donors and would
have had to pass the same reachability gate at the selected layer before any
report ran. The reporter itself is the existing 24-cell episode-remapped `Q/K`
interface, plus a new visible capability control described below.

## Result

The reachability gate failed at every anchor, so **no reporting row ran**.

| gate | frozen requirement | L9 | L21 | L26 |
|---|---:|---:|---:|---:|
| clean unrestricted answers | 10/10 | **10/10** | 10/10 | 10/10 |
| exact self-patch | ≤ 1e-4 | **0.0** | 0.0 | 0.0 |
| bidirectional cross-patch | ≥ 4/5 tasks | **0/5** | **0/5** | **0/5** |
| mean normalized recovery | ≥ 0.5 | **+0.001** | **−0.003** | **+0.100** |

The task itself is now reliable, which the route task never was: all ten clean
answers are correct at conditional probability 1.000, against 8/10 before. So the
failure is in the transplant arm, not in the model's competence at the task.

No cross-patch changed the top-1 answer at any layer. Recovery rises
monotonically with depth, and the only two cells above 0.11 are at layer 26
(`1+1|2` at 0.351 and `9-9|8` at 0.435, both patching the odd recipient with the
even donor). Under the frozen estimand that is a failed gate, not a weak
positive.

## The plumbing is not the explanation

Two independent checks run on every patch. The self-patch — replacing a state
with itself — reproduces the full next-token logit vector with maximum absolute
error **0.0** at all three layers, and the hook re-reads the post-block residual
and refuses to continue unless it equals the donor tensor. The write happens, it
is exact, and the answer still does not move.

What the plumbing check cannot rule out is that the two twin states are so alike
at these layers that replacing one with the other is barely an edit. No pair has
a matching state hash at any layer, but that is a weak reassurance. The
diagnostic below measures it properly, and the answer turns out to be the whole
story.

## Where the answer actually is

**Post-hoc, development bank only, no reporting claim.** The frozen run tests
three anchors; it cannot say whether the site exists elsewhere, which is exactly
what `09` said to establish before spending another report bank.
`scripts/diagnose_answer_site.py` measures three things at the same position over
all 36 layers: the same twin transplant, the relative size of the edit it makes,
and a logit-lens read of the clean state through the final norm and unembedding.
The third separates the two live explanations — a state that carries the answer
but is overwritten by downstream recomputation, versus a state that does not
carry the answer yet at any layer tested.

Whatever it returns cannot turn this run into a positive, and a layer chosen from
it cannot be used to run the reporter under this protocol. Its only job is to say
whether the family is worth a fresh pre-registration or is finished.

It is worth one. Every measurement changes at the same place, between blocks 26
and 27 — 78% of the way through a 36-block model, and one block past the deepest
anchor the protocol named.

| after block | recovery | donor answer top-1 | ‖donor − recipient‖ / ‖recipient‖ | lens margin | lens sign correct |
|---:|---:|---:|---:|---:|---:|
| 9 | +0.001 | 0/10 | 0.03 | −0.02 | 0.5 |
| 21 | −0.003 | 0/10 | 0.13 | +0.11 | 0.5 |
| 26 | +0.100 | 0/10 | 0.23 | +0.31 | 0.5 |
| **27** | **+0.787** | **10/10** | **0.49** | **+1.48** | **0.8** |
| 29 | +0.783 | 10/10 | 0.54 | +6.55 | 1.0 |
| 33 | +0.797 | 10/10 | 0.54 | +9.52 | 1.0 |
| 35 | +1.000 | 10/10 | 0.85 | +19.00 | 1.0 |

The last row is an identity, not a measurement: patching after the final block
sets the state the output head reads, so recovery is 1.000 by construction. It is
the arithmetic check that the estimand is scaled correctly.

Read the lens column first. Through block 26 the clean state does not favour its
own answer over its twin's at better than chance — the margin's sign is right on
4 to 6 of 10 problems at every layer down there, and never more. The two twin
states also stay close together, under a quarter of the residual norm apart at
every block below 27. **There was nothing to transplant.**
The model has not computed the answer at the answer position yet, so exchanging
that state exchanges almost nothing, and the ordinary answer correctly does not
move.

From block 27 the same single-position transplant controls the answer completely:
the donor's digit is the full-vocabulary argmax in 10 of 10 transplants, which is
5/5 tasks in both directions, at a mean recovery of 0.787. Three of the gate's
four criteria are therefore met at block 27, and the fourth — exact self-patching
— was 0.0 at all three anchors and does not depend on depth. Had 27 been on the
anchor list, this pilot would in all likelihood have gone on to run the reporter.

So the interpretation is not that this model's naturally computed states resist
transplanting. It is that at the last pre-answer token this model computes its
answer in a single narrow band late in the stack, and the three anchors — chosen
before the run from the layers earlier results had used — all sit below it.

## What follows

**Observation:** on a task the model solves perfectly, exactly replacing the
residual at the position that produces the answer does not make the answer
follow, at three prospectively named layers — because the answer is not there
yet. One block deeper, the same intervention controls the answer in 10/10
transplants.

**Interpretation:** an instrument result about where to intervene, not evidence
about introspective reporting. The reporter was never evaluated, so nothing here
bounds the reporting capability that [`06`](06-causal-codebook-icl.md) and
[`07`](07-trained-activation-reporter.md) measured on injected directions.

**What this negative is worth:** a caution about anchor placement, and the
anchors were not chosen carelessly — 9 is the layer every prior reporting result
in this repository used, and 21 and 26 are the sites the Stage 1b screen selected
for the codebook effect. None of that transfers. Where a *planted* direction is
readable says nothing about where a *computed* answer lives, and the gap here is
the difference between a failed experiment and a passed one.

**Closed in scope:** this protocol is finished. Its verdict stands as recorded,
and no reporting number may be quoted from it. In particular, do not run the
reporter at block 27 under this protocol and call the result pre-registered — the
layer was chosen after seeing data, which is exactly what the frozen anchor list
existed to prevent.

**Reopen condition, met on development data.** [`09`](09-natural-state-pilot.md)
required a site whose cross-patch changes the ordinary answer, established on a
disjoint development bank under a reliable clean task, before any fresh report
bank is exposed. Blocks 27 through 34 satisfy that on the development bank with
clean answers 10/10 and cross-patching 5/5 — block 35 is excluded as the identity
case above. The successor was therefore narrow: freeze block 27 as the site,
confirm reachability on the held-out bank, and run the reporter and its visible
capability control once. It ran the same day under its own protocol, and the
held-out confirmation failed; the section below records it.

One thing to expect from it, and to say in advance rather than after: at block 27
the state's own answer is already legible to a logit lens. A positive would show
the interface reading a state the model computed for itself, which no result here
has yet done, but the property being reported is close to the answer token. That
is the narrow claim this design was chosen to buy, not a surprise to be explained
once the number exists.

## That successor ran, and the held-out bank did not confirm

`natural_state_arith_l27_smoke_protocol_v1.json` froze block 27 as a **disclosed
post-hoc selection**, changed nothing else — same stimuli, banks, conditions,
gates and stop rule — and required the site to pass the same reachability gate on
the held-out bank before any reporting row. The held-out bank had never been
scored, by the screen or the diagnostic.

| bank | clean | self-patch | cross-patch tasks | mean recovery | gate |
|---|---:|---:|---:|---:|---:|
| development | 10/10 | 0.0 | **5/5** | +0.787 | pass |
| held-out | 10/10 | 0.0 | **3/5** | +0.697 | **fail** |

The development bank reproduced the diagnostic exactly. The held-out bank did
not, so **the reporter did not run**, and no reselection is permitted under this
protocol.

The failure is narrow and one-sided. Eight of ten held-out transplants worked, two
did not, and both failures are the same direction — patching an even-answer
recipient with its odd-answer twin, in `6+2|3` (donor answer reached p=0.469, a
coin flip) and `7-3|2` (p=0.029). Every odd-recipient transplant succeeded, most
at p≥0.99. The recipient's own answer sometimes re-asserts itself.

**The gate had less power than it looked.** Pooling both banks, 18 of 20
transplants succeed, so the per-transplant rate is about 0.90. A task passes only
if both of its directions do, ~0.81, and "at least 4 of 5 tasks" then succeeds
about 76% of the time even when nothing is wrong. At a per-transplant rate of
0.85 it drops to about 58%. With five tasks the criterion moves in 20% steps and
cannot distinguish 0.90 from 0.75. The development bank's 5/5 was therefore not
strong evidence that the held-out bank would pass, and the honest reading of the
pair is one effect of roughly 0.9 measured twice, not a replication failure.

That is a criticism of the frozen design, made after seeing it fail, and it does
not license reading the result as a pass. What it licenses is the next
protocol's bank size.

## Where this leaves the branch

Gate 3 — a causally reachable natural state — is **not cleared**. It is closer
than it was: the site is located, it survives on development data, and the
instrument is exact. What is missing is a transplant reliable enough to carry a
report bank, and the two candidate repairs are visible in the data rather than
speculative:

- **More tasks.** Ten or twenty twin pairs instead of five, so the gate measures
  the transplant rate instead of quantising it. This is the cheap one, and the
  power calculation above says it is the one that was actually wrong.
- **More than one block.** Recovery is 0.78–0.80 across blocks 27–33 on
  development data and never reaches 1.0 short of the final block, which is what
  a partially re-asserted computation looks like. Transplanting the donor's
  column across a band of blocks at that one token, rather than a single block,
  is the intervention that matches the mechanism.

Both need a fresh protocol. Neither may reuse the held-out bank that has now been
scored: it has been spent, and a third bank has to be written before either runs.

## The reporter finally ran, and the capability control caught it

`natural_report_l27_protocol_v1.json` changed the unit of certification and
nothing else. Twelve fresh pairs, problems disjoint from both earlier banks; each
pair certified on its own by requiring both of its transplants to carry the
ordinary answer; the reporter then run on the first five certified pairs **in
frozen bank order, not the five with the largest effect**.

The certification worked exactly as the power arithmetic predicted. **Nine of
twelve pairs certified**, against an expectation of 9.7 at the 0.81 per-pair rate
the earlier runs implied. Every clean answer was correct, and every self-patch
was exact. For the first time in this family, the reporter ran: 120 episodes,
five conditions plus the visible control.

| metric | frozen gate | result |
|---|---:|---:|
| natural accuracy | ≥ 0.75 | **0.500** |
| natural − query-only | ≥ 0.20 | **0.000** |
| natural query-twin both correct | ≥ 0.60 | **0.000** |
| anti-grounded inverse | ≥ 0.75 | 0.500 |
| next-token format rate | ≥ 0.90 | 1.000 |
| mean label mass | ≥ 0.50 | 1.000 |
| sham matches clean | ≤ 1e-4 | **0.000** |
| **visible capability** | ≥ 0.75 | **0.533** |

**This is not a reporting null.** The frozen interpretation gate says so, and it
is the reason that gate exists. At 0.533 the model cannot induce the parity rule
*with the arithmetic written out in plain text and nothing patched at all*. A
0.500 under transplanted states cannot be attributed to state reading when the
same episodes with visible content also sit at chance.

The failure signature is worth recording precisely. Natural accuracy is exactly
0.500 on all five pairs, query-only is exactly 0.500, mapping-flip pairs are
0.500 and query-twin pairs are 0.000. A twin-pair score of 0.000 with 0.500 row
accuracy means one thing: the model emits the same label whichever state is in
the query slot. It is answering from the visible episode structure, which is the
prompt-only strategy the twin-pair statistic was built to expose. Meanwhile
format is 1.000, label mass 1.000, and the sham arm reproduces clean to 0.000 —
the instrument is in perfect health and is reading nothing.

## What actually blocks this, after five runs

The three candidate blockers can now be separated, because each was measured on
its own:

| component | evidence | verdict |
|---|---|---|
| the transplant | 9/12 pairs certified bidirectionally; self-patch exact | **works** |
| the interface | 0.891 on injected directions; format 1.000, sham 0.000 here | **works** |
| the hidden class | visible control 0.533 with no patching at all | **fails** |

So the blocker is not the site, not the plumbing, and not the reporter. It is
that `Qwen2.5-3B-Instruct` cannot learn "even result → Q, odd result → K" from
four demonstrations.

That also bounds the result this branch is built on. In
[`06`](06-causal-codebook-icl.md) the hidden classes are one direction and its
negation — a single axis with two poles, which four demonstrations can separate
without inducing anything. Here the classes are five unrelated computed states
sharing an abstract property. The 0.891 is evidence of reading a planted axis; it
is not evidence of reading semantic content, and this run is the first direct
probe of that distinction. It sits comfortably with
[`08`](08-sensitivity-specificity-tradeoff.md), where training pushed the reader
to 0.913–0.955 on random directions: axis detection, not semantics.

**The next design must change the hidden class, and the visible control is the
screen for choosing it** — it needs no patching, no site, and no model
intervention, so candidate classes can be screened for inducibility before any
transplant work is done at all. That inverts the order of the last five runs,
which spent four of them on the intervention and discovered the class was
unusable on the fifth.

## The visible capability control, which earned its place

The reporter was gated behind reachability and never ran, but its capability
control is implemented and worth recording, because it is what would have made a
reporting null interpretable. It presents the identical induction problem with
the arithmetic written out and nothing patched: four demonstrations labelled `Q`
or `K` by the parity of their result, then a held-out problem. The frozen rule is
that a natural-state null counts as a reporting null only if visible accuracy is
at least 0.75; below that the model cannot induce the rule at all and the null is
an instrument failure. The route pilot had no equivalent, which is one reason its
null would have been hard to read.

## Artifacts

Blind anchor screen (9, 21, 26):

- `results/natural_state_arith_smoke_protocol_v1.json`,
  `results/natural_state_arith_smoke_v1_raw.jsonl`,
  `results/natural_state_arith_smoke_v1_raw.manifest.json`,
  `results/natural_state_arith_smoke_v1_summary.json`.

Post-hoc localization diagnostic, development bank only:

- `results/natural_state_arith_site_diagnostic_v1.json`.

Named-site confirmation at block 27:

- `results/natural_state_arith_l27_smoke_protocol_v1.json`,
  `results/natural_state_arith_l27_smoke_v1_raw.jsonl`,
  `results/natural_state_arith_l27_smoke_v1_raw.manifest.json`,
  `results/natural_state_arith_l27_smoke_v1_summary.json`.

Source: `scripts/run_natural_state.py`, `src/introspect/natural_state.py`,
`scripts/diagnose_answer_site.py`.

Both runs are named `smoke` and are nonetheless the citable runs for their
verdicts. `--smoke` truncates the reporter from 24 cells to 4 and is read nowhere
before the reporter; donor preparation, the clean gate, and the reachability
screen are byte-identical in both modes. Since the reporter never ran in either,
a full-mode run would produce the same screen and the same stop, and neither was
re-run to relabel it.

The `--site` option and its protocol branch were added to the runner after the
blind screen was frozen, so `natural_state_arith_smoke_protocol_v1.json` records
a `scripts/run_natural_state.py` hash that no longer matches the tree. The
frozen file is the record of what executed; `git log` holds that source. Every
other source hash in it still matches.
