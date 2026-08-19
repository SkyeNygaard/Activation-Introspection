# Plan: four free checks, two of which could kill the result

Written **2026-08-18**, before running any of them. All four re-use data already on
disk; none needs the model. Total cost: minutes.

## Why these four

The result so far is that comparing two models' internals ranks the damaged
questions better than comparing their outputs. Before spending anything on scale or
on causal follow-ups, the cheapest useful thing is to try to break it. Two of these
four could.

### 1. Are some questions just fragile? *(could kill it)*

Three different fine-tunes were tested on the same 300 questions. If the same
questions get damaged every time, then an auditor needs no second model and no
internals at all — they just need a list of questions that break easily, which they
could get from any previous fine-tune.

**What I do:** measure how far the damage under one fine-tune predicts the damage
under another, and use it as a ranking signal against the internal one.

**Outcomes.** If fragility ranks as well as internals, the white-box story is
undercut and the honest headline changes. If fragility ranks near zero, the damage
is specific to each fine-tune and the internal signal is doing something a question
list cannot.

### 2. Is the damage measure partly an artifact of where the model started? *(could kill it)*

Damage is defined as *base score minus tuned score*. A question the untouched model
answers at 95 has more room to fall than one it answers at 60. So high starting
scores mechanically permit bigger drops, and if the internal signal happens to track
the starting score, some of the result is that arithmetic rather than the fine-tune.

**What I do:** measure whether the untouched model's own score predicts the drop,
and re-run the comparison with that removed from both sides.

**Outcomes.** If the gap between internals and outputs survives, the result stands
as stated. If it does not, the ground truth needs redefining and every number so far
is provisional.

### 3. Does a reader calibrated on one fine-tune flag a different one?

Everything measured so far fits a reader using damage labels *from the fine-tune it
is being tested on*. That is a real workflow — judge a sample, rank the rest — but it
is not the strong version. The strong version is: calibrate on a fine-tune you
already know is bad, then point it at one you have never seen.

**Outcomes.** If it transfers, this is an auditing tool rather than a measurement,
and it is the most useful thing in the project. If it does not, the method needs
labels for the very thing it is meant to find, and that limit has to be stated
plainly rather than left for a reader to notice.

### 4. Did the three readers find the same direction?

If the directions the three fitted readers use point the same way, that is one shared
"this fine-tune is engaged" axis, which would explain a transfer result and connects
to published work reporting a single direction behind this kind of misalignment. If
they point in unrelated directions, transfer should fail, and the reader is learning
something specific to each fine-tune.

This is a prediction that ties 3 and 4 together: **whatever 4 says, 3 should agree
with it.** If they disagree, I have misunderstood something and will say so.

## Also worth having, while the data is open

A harder target: rank against the questions that cross the published *below thirty
out of a hundred* bar — outright bad answers rather than mild degradation. There are
32, 19 and 30 of them. An auditor cares more about those than about a question that
slipped from 88 to 75.

---

## Follow-up planned 2026-08-18, after the four checks

The checks overturned the headline: a list of which questions broke under a
*different* fine-tune ranks better than anything read off the models, and internals
add nothing on top of it. Two things follow, and both are worth their cost.

### A. Is anything able to predict the part the list gets wrong? *(free)*

The damage on a question splits into a part shared with other fine-tunes and a part
specific to this one. The list captures the shared part by construction. **Does
anything predict the specific part?**

If internals do, they have a narrow but real job: catching what a fine-tune broke
that its siblings did not. If nothing does, the fine-tune-specific part is not
predictable from the question at all, which closes the line honestly and is worth
saying plainly.

### B. Does the fragile-question list transfer to a different model family? *(~1 hour)*

This is the difference between a curiosity and a usable artifact. If the same
questions are fragile in `Qwen2.5-0.5B-Instruct` as in `Llama-3.2-1B-Instruct` —
different families, different training data, different sizes — then the list is a
property of the *questions* and can be measured once and reused by anyone. If it
does not transfer, the list only works within a model you have already audited,
which is a much weaker claim and re-opens room for methods that read the model in
front of you.

Qwen2.5-0.5B was rejected earlier as too incoherent to be the main platform. It is
acceptable *here* because the question is only whether the damage ranking agrees,
and incoherent answers are discarded before ranking. Expect to lose more questions
than at 1B and to say so.

**Cost:** three adapters that are already published, about fifteen minutes of
generation each at this size, plus judging. No training.

**The boring outcome is informative.** If the lists do not correlate at all, the
result becomes "fragility is real but model-specific", which is a smaller claim
honestly stated rather than a failure.
