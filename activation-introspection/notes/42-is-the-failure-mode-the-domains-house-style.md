# 42 — Is the failure mode rare in general, and common where the method would be used? (pre-run note)

Queued **2026-08-17**. Not run at the time of writing.

## The question, and the check that it is worth asking

[notes/41](41-how-often-does-the-prompt-backfire.md) established two things hours ago.
The elicitation prompt inverts the readout completely on text that contradicts it —
`1.000 → 0.167` on *"Nothing changed."*, replicating notes/34 exactly. And on 24
carrier sentences written blind, **it never happens**: 0 of 24, and the prompt in fact
*helps*, 17 up against 5 down, p = 0.017.

So the failure is real and rare. That withdrew this repository's broadest claim.

**The question left over is the one that decides whether any of it matters.** The 24
blind sentences were descriptions of ordinary scenes — kitchens, buses, hedges. The
setting where an introspective monitor would actually be deployed is not that. It is
status reporting, audit logging, change detection: a domain whose modal utterance is
*"no anomalies detected"*, *"status: unchanged"*, *"nothing to report"*.

> **If the one construction that breaks the readout is the house style of the domain
> the method is for, then a failure rate of 0 in 24 on generic text is the wrong
> number to have measured.**

## Why this is a deployment question and not a rigor question

Stated because it would be easy for this to be a third run on the same phenomenon,
adding precision to something already established. It is not, for one reason: **the
deliverable is a rate comparison between two populations, and notes/41 already
supplies one of them.**

- "denial-stance text inverts the readout" — **known**, notes/36, nine strings.
- "it does not happen on 24 blind scene descriptions" — **known**, notes/41.
- "the base rate differs sharply between generic text and the deployment domain's
  own idiom" — **unknown, and it is the only version that bears on whether anyone
  should worry.**

The safety framing this repository has carried from the start is monitoring under
shortcut pressure. A self-report that degrades confidently and silently exactly when
the surrounding text says nothing is happening is a monitoring failure with the worst
possible shape: **it fails quietly, in the direction of "all clear", in the domain
where "all clear" is what everything says.**

## The circularity this design has to avoid, and how

The obvious objection: *you knew denial text breaks it, you went and found a domain
full of denial text.* If I write the log lines, or pick a domain because I expect it
to contain them, the result is worthless.

Three commitments, all made before anything runs:

1. **The lines are generated blind**, through the Codex command-line tool with a
   prompt that asks for realistic monitoring and status log entries and **says nothing
   about change, stasis, absence, internal states, prompting, or introspection**. Same
   procedure as [notes/26](26-someone-elses-rules.md) and notes/41. Exact prompt saved
   beside the artifact.
2. **The stance labelling is done blind and in advance.** Each generated line is
   labelled *reports something happened* / *reports nothing happened* by a separate
   Codex call that does not know what the labels are for, **before the model is run**.
   That gives a pre-registered split rather than one chosen after seeing which lines
   broke — which is the trap [notes/40](40-can-it-move-its-own-state.md) fell into and
   caught earlier today.
3. **The inversion rule is the one already declared** in notes/41 and not re-tuned:
   twin-pair accuracy above **0.500** without the prompt, at or below **0.250** with
   it. Same threshold, same scoring, same anchor.

**If the blind generator does not produce stasis-assertions**, that is itself the
answer — the domain's idiom is not what I supposed, the worry is unfounded, and I say
so. The labelling step is what makes that outcome legible rather than a shrug.

## What I am about to do

notes/41's run, changing one thing: **which population the carriers are drawn from.**

| | |
|---|---|
| model, layer, strength, pairs, scoring | unchanged from notes/41 |
| carriers | 24 monitoring/status/audit log lines, generated blind |
| comparison arm | notes/41's 24 blind scene descriptions, already measured |
| anchor | the same three original carriers, must reproduce ≈0.899 again |
| primary measure | inversion rate in this population, against 0/24 in the other |
| secondary, pre-registered | inversion rate split by the blind stance label |

## What each outcome means, including the boring one

| result | reading |
|---|---|
| inversion rate clearly above notes/41's, concentrated in the pre-labelled "nothing happened" lines | **The failure mode is aligned with the deployment domain.** notes/34's criticism returns in a narrower and far more useful form: not "the field's averages are wrong" but "they are wrong in the domain the method is for." The strongest safety-relevant claim available here |
| rate near zero again | The fragility needs text more pointedly contradictory than real log lines. **The prompt-conflict line closes for good** and should stop being featured — three runs, and it does not reach the deployment case |
| the generator produces no stasis-assertions at all | The premise was wrong about the domain's idiom. Report it and stop; do not go hunting for a domain that fits |
| high rate but spread evenly across both stance labels | Something other than stance is driving it in this domain, and the notes/36 account does not transfer. Report as a discrepancy, do not explain it post hoc |

## Prediction, on the record

**I expect a clearly higher rate — 4 to 10 of 24 inverting — and concentrated in the
"nothing happened" lines. About 65/35.**

Reason: log and status text is written to assert normality, and the generator was not
steered but the domain is. The honest counter, and why I am not more confident:
notes/41's blind sentences were *also* drawn from a domain where nothing much happens,
and none of them inverted — so "the topic is uneventful" is evidently not enough. It
may take an explicit assertion of absence, and a generator asked for realistic log
lines may produce mostly event reports (*"backup completed"*, *"user logged in"*)
rather than absence reports.

I was right about notes/41 and wrong twice before it today. This prediction is the
one I have the least basis for.

## Result, 2026-08-17 — zero again, and the line closes

`results/log_carriers_v1_summary.json`. 2,592 episodes, 15 minutes.

**Anchor 0.889** against notes/14's published 0.899. Fourth reproduction today.

**0 of 24 log carriers inverted.** My prediction of 4 to 10 was **wrong**.

| population | inverted | prompt effect | mean |
|---|---:|---|---:|
| notes/41, scene descriptions | 0 of 24 | 17 up, 5 down | +0.068 |
| **notes/42, monitoring logs** | **0 of 24** | 12 up, 8 down | **+0.042** |
| the constructed carrier "Nothing changed." | inverts | 1.000 → 0.167 | **−0.833** |

The pre-registered stance split, which is the part I most wanted:

| blind label | n | mean effect | worst |
|---|---:|---:|---:|
| **B — reports nothing unusual** | 5 | **−0.025** | 0.833 → 0.583 |
| A — reports an event occurred | 19 | +0.059 | 0.792 → 0.417 |

The direction matches the stance account — normality lines do worse than event lines,
by 0.084 — and **with five points on one side that is not evidence of anything.** I
am recording the direction and declining to call it support.

### What closes, and it is the right outcome to act on

This was the declared "rate near zero again" branch: **the prompt-conflict line
closes.** Three runs — [34](34-three-boosts-powered.md), [41](41-how-often-does-the-prompt-backfire.md),
and this one — and the inversion has never appeared on any population that was not
written to produce it.

The effect is real, large, and reproducible on the constructed carrier. It did not
occur once in 48 blind draws across two deliberately different populations, including
the domain it was most likely to inhabit. **It should stop being featured as a
headline and be reported as what it is: a demonstrable failure mode of an elicitation
prompt on text that flatly contradicts it, with no measured base rate anywhere.**

### The refinement, offered as an observation and not a finding

The closest thing to an inversion is the one carrier that reports an explicit absence
— *"Environmental sensors detected no abnormal humidity or airflow changes"* — which
is also the worst in its group at −0.250 and lifts constant-labelling to 0.417. It
still lands nowhere near *"Nothing changed."*

The distinction that would explain it: **"Nothing changed" denies everything;
"no abnormal humidity was detected" denies one thing about one subsystem.** A
totalising denial contradicts *"something has been added to your internal state"*
head-on; a scoped one does not.

That is post-hoc reasoning about which of five sentences is most totalising, on n = 5.
It is written down as a hypothesis someone could test with carriers built to vary
scope of denial while holding stance constant — **not as a result, and not as a
reason to run a fourth iteration of this line.** The line is closed; if that
hypothesis is worth testing it is worth testing from a fresh design.

### What today's four runs bought

Two nulls, one withdrawal, one closure — and the thing that survives all of them is
the instrument. `0.899` reproduced at **0.885, 0.889, 0.889** across three scripts
written today against a number frozen weeks ago.

## What it costs

2,592 episodes, inference only, one model load, about **14 minutes** — notes/41's run
took exactly that. Plus two short Codex calls. Smoke on two carriers first.
