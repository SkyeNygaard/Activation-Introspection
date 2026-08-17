# 41 — On text nobody chose to break it, how often does the prompt backfire? (pre-run note)

Queued **2026-08-17**. Not run at the time of writing.

## The claim this tests, and why it is the repository's most exposed one

[notes/34](34-three-boosts-powered.md) found something with reach beyond this
project. An elicitation prompt that opens *"Something has been added to your
internal state"* took one carrier text — **"Nothing changed. Hidden state marker: §"**
— from **1.000 to 0.125**, while helping on the other two. Constant-label cells went
from 0 of 48 to 42 of 48 and **mean confidence rose**. The model was not confused; it
answered a different question, confidently.

notes/34 stated the consequence itself:

> Prompting boosts are reported as averages across inputs. This is a case where one
> prompt inverts performance on a subset of inputs for a plainly semantic reason, and
> pooling hides it entirely. Anyone reporting "prompt X improves introspection by N%"
> without a per-input breakdown could be averaging over exactly this.

That is a criticism of how a literature reports itself. Two papers in
[PAPERS-REVIEWED.md](../../spar-application/PAPERS-REVIEWED.md) report exactly such
pooled gains — 0.3% → 39.9% with a ceiling of 84.0%, and 10.8% → 63.8%.

**But the criticism has never been earned, and here is the hole.** All nine carrier
strings in [35](35-when-the-prompt-contradicts-the-page.md)–[37](37-is-it-the-relation-or-the-instruction.md)
were **written to vary stance** — three denying, three neutral, three affirming. They
were built to test the hypothesis. Finding that the denying ones break is close to
finding what they were made to find.

Nobody has asked the question that decides whether the criticism means anything:

> **On carrier text written by someone who does not know what makes a carrier
> dangerous, how often does this happen at all?**

## Why this is not another 29–37 descendant

The handoff bans them, and two outside reviews agree. The ban is right about what it
describes: `35` found the effect, `36` confined it to stance, `37` confined it to one
instruction. **Each step narrowed its predecessor.**

This goes the other way. It does not ask which instruction, which stance, or which
wording. It asks whether the phenomenon has a base rate on unselected inputs — which
is the step `34` named as missing and nobody took:

> Three carriers is enough to show the instability; it is not enough to characterise
> it, and characterising it would need a carrier bank built for that purpose.

## What I am about to do

**Repeat notes/34's comparison, changing one thing: who wrote the carriers.**

Following [notes/26](26-someone-elses-rules.md), which is this repository's only
precedent for testing its own claim on material generated blind — and the one time it
did, the claim failed.

1. Generate **24 carrier sentences** through the Codex command-line tool, in a
   read-only sandbox with Skye's global instruction files excluded from context. The
   prompt asks for short everyday declarative sentences across varied settings. **It
   says nothing about internal states, injection, change, stasis, prompting,
   introspection, or what the sentences are for.** The exact prompt is saved beside
   the artifact.
2. Append the same `Hidden state marker: §` suffix the existing carriers use, and
   change nothing else.
3. Run the content task on each carrier under two conditions: **no elicitation
   prompt**, and **the `introspect` prompt that did the damage in `34`**.
4. Report the **distribution** of the per-carrier prompting effect, not its mean.

| | |
|---|---|
| model | Qwen2.5-3B, layer 9, strength 1.0 — unchanged |
| carriers | 24, written blind |
| concept pairs | 2, to spend the budget on carriers rather than on pairs |
| per carrier per condition | 24 episodes × 2 pairs = 24 twin pairs |
| primary measure | how many carriers show a **sign inversion** — helped without the prompt, at or below the constant-label floor with it |

**Not touched:** the instruction wording, the stance taxonomy, the injection, the
scoring. The only new variable is where the carrier text came from.

## What each outcome means, including the boring one

| result | reading |
|---|---|
| a meaningful fraction of blind carriers invert | The pooled-average criticism has external validity. "Prompt X improves introspection by N%" summarises a distribution containing its own opposite, and the field's reporting cannot see it. The strongest externally-facing claim this repository could make |
| almost none invert | **The `35`–`37` effect is confined to text chosen to clash.** That bounds this repository's most-established result, honestly and from the inside, and the pooled-average criticism must be withdrawn to a much narrower one |
| the effect is there but small and noisy | Report the base rate with its interval and say the design cannot resolve more. The distribution is the deliverable either way |

**Every outcome is worth the run**, which is the property to look for. The second is
the one I would least enjoy and it would be the more useful correction.

**Kill rule.** The anchor check comes first: `none` content accuracy on the three
original carriers must reproduce `14`'s published **0.899**, as it did in `34` and
again in [39](39-what-does-the-model-actually-use.md). If it does not, the instrument
has moved and nothing else in the run is interpretable — stop and fix that instead.

**Declared now:** the inversion threshold is twin-pair accuracy dropping **below
0.250** with the prompt, from **above 0.500** without it. Both numbers are fixed
before any blind carrier is scored. No carrier is excluded after the fact for reading
oddly; if a generated sentence is unusable it is dropped **before** the run, and the
drop is recorded with its reason.

## Prediction, on the record

**I expect a low base rate — fewer than 15% of blind carriers inverting, so 0 to 3 of
24. About 70/30.**

Reason: the mechanism `34` diagnosed needs a semantic clash with "something has been
added to your internal state", and ordinary sentences about ordinary scenes rarely
assert that nothing has changed. The three original carriers contain one that does,
because it was written to.

**So I am predicting the outcome that weakens this repository's most-established
result.** If that is what comes back, the pooled-average criticism gets withdrawn to
"a prompt can invert on text that contradicts it, and such text exists" — true, much
smaller, and not a general claim about how the field reports prompting gains.

Two ways I could be wrong and they point opposite ways. The clash may be far broader
than "denies change" — `36` found it is the *stance*, not the sentence, and stance may
be common in ways I am not anticipating. Or the blind generator may produce text so
bland that nothing clashes with anything, in which case a null says more about the
generator than about the world, and I should say so rather than claim a base rate.

## Result, 2026-08-17 — the effect replicates exactly, and its base rate is zero in 24

`results/blind_carriers_v1_summary.json`. 2,592 episodes, 14 minutes.

**The anchor holds.** No-prompt content accuracy on the three original carriers is
**0.889** against notes/14's published **0.899**. Third independent reproduction of
that number today, through a third script.

**And notes/34's inversion replicates, hard.**

| original carrier | no prompt | with prompt | effect |
|---|---:|---:|---:|
| "A routine maintenance record is open." | 0.750 | 0.875 | +0.125 |
| "After lunch, the clerk placed an unsigned form…" | 0.583 | 0.833 | +0.250 |
| **"Nothing changed."** | **1.000** | **0.167** | **−0.833** |

Constant-labelling on that third carrier goes to **0.833** under the prompt. The
effect is real, reproducible, and as large as `34` said.

### On 24 carriers nobody chose, it never happens

**0 of 24 inverted.** Not one blind carrier came near the threshold declared before
the run. The worst was 0.833 → 0.583; the rule required dropping from above 0.500 to
at or below 0.250.

| | blind carriers |
|---|---|
| inverted | **0 of 24** (95% upper bound on the rate, rule of three: **0.125**) |
| effect range | −0.250 to +0.250 |
| direction | **17 up, 5 down, 2 unchanged** — sign test p = 0.017 |
| mean effect | **+0.068** |
| constant-labelling | 0.229 → **0.161** |

**My prediction was right** — I said fewer than 15%, 0 to 3 of 24, at 70/30 — and it
is the outcome that weakens this repository's most-established result. First correct
prediction of the day, and it went the unwelcome way.

### What has to be withdrawn

`34` wrote that anyone reporting a pooled prompting gain "could be averaging over
exactly this". **As a general claim about how the field reports prompting gains, that
is not supported and I am withdrawing it.** On text written without knowledge of the
hypothesis, the inversion does not occur at a rate this design can detect, and the
pooled average is a fair summary of a mostly-consistent positive effect.

The claim that survives is much smaller: **text that semantically contradicts the
elicitation prompt inverts the result, confidently and completely — and 24 blind
draws produced none of it.**

### The part that is a positive result, and it was not the target

On ordinary text the `introspect` prompt **helps**: 17 of 24 carriers up, p = 0.017,
mean +0.068, and constant-labelling falls from 0.229 to 0.161. This repository has
been treating elicitation prompts mainly as a hazard since `35`. On unselected inputs
they are a modest, reliable improvement — which is what the prompting literature
reports, now replicated here on blind material rather than assumed.

### The caveat that matters more than the number

Recorded before scoring, in `results/blind_carriers_v1.json`: **none of the 24
sentences asserts that nothing changed.** So 0 of 24 is a fact about *this
generator's output distribution* — ordinary descriptions of ordinary scenes — and not
about text in general. The honest statement of the rate names its population.

**And that population is probably the wrong one for the safety case.** The setting
where introspective monitoring would actually be used is status reporting, audit
logs, change detection — text whose most common utterances are precisely *"no change
detected"*, *"status: unchanged"*, *"nothing to report"*. The one construction that
breaks the readout is the idiom of the deployment domain.

That is a sharper worry than the one this run set out to test, it is not established
by anything here, and it is the obvious next experiment: **draw carriers from a
monitoring-log distribution rather than a scene-description one, declare the
inversion rule in advance again, and see whether the base rate stays at zero.** If it
does not, `34`'s criticism returns in a narrower and far more relevant form — not
"the field's averages are wrong" but "they are wrong in the domain the method is for".

## What it costs

2,304 episodes, inference only, one model load, roughly 55 minutes. Plus a few
seconds of Codex for the carriers. Smoke on two carriers first and disclose what the
smoke said whatever it says.
