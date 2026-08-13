# 32 — Does boosting introspection cost selectivity? (pre-run note)

Written before anything ran.

## The gap this walks into

[Macar et al.](https://arxiv.org/abs/2603.21396) — the current frontier — report
that introspection is **substantially underelicited**, and that you can fix it
without training. Ablating the refusal direction lifts detection from 10.8% to
63.8%. They report false positives rising only from 0.0% to 7.3% and call the
capability "underelicited by default".

**Their false-positive control is unsteered trials: nothing injected at all.** It
asks whether the model claims to feel something when nothing happened.

That is not the control that matters for a monitor. [Lindsey's
original](https://arxiv.org/abs/2601.01828) established the harder one and showed
base models pass it: **norm-matched random vectors** need a larger norm to be
noticed at all, and even then reach only 9 trials in 100. Something real happened;
it just meant nothing. An untrained model can tell the difference.

Nobody has run that control on a refusal-ablated model. Macar et al.'s own
Responsible Use section asks for exactly this and does not do it: *"Methods that
boost introspection should include side-effect audits."*

## Why this is worth a run now rather than a proposal

[`08`](08-sensitivity-specificity-tradeoff.md) established that one way of boosting
introspection — **training** — destroys selectivity: magnitude-matched random
directions go from 0.513 to 0.913–0.955.
[`29`](29-can-abstention-recover-selectivity.md) showed abstention cannot repair
that, and [`31`](31-why-training-inverts-abstention.md) found why: filtering for
confidence *enriches* for confidently-labelled noise, with accuracy on random
directions climbing to 0.98.

So the question is no longer "does boosting cost precision". It is sharper:

> **Is the selectivity collapse a property of training, or of boosting
> introspection by any means?**

Refusal ablation is the test case, because it is the cheapest boost, it requires
**no training** — which the standing no-further-LoRA decision makes necessary, not
merely convenient — and it is the one the frontier paper endorses.

## What I am about to do

**Step 1 — build the refusal direction**, by Arditi et al.'s difference-in-means:
mean residual activation on harmful instructions minus mean on harmless ones, per
layer, at the post-instruction position. One candidate layer is selected on a
held-out split by which direction most reduces refusal on harmful prompts, and
frozen before any introspection episode runs.

**Step 2 — ablate it**, again by their method: zero the component along that
direction at **every** layer and position, `x ← x − r̂ r̂ᵀx`. This is the
"abliteration" Macar et al. use.

**Step 3 — rerun [`14`](14-content-versus-disturbance.md)'s design unchanged**,
with and without ablation. That design is the right one because it already
contains the control at issue: two different concepts against **two random
directions at identical class separation by construction**, with byte-identical
visible text and twin-pair scoring. Published baseline: content 0.899 row /
0.799 twin-pair, random 0.594 / 0.188.

Four numbers decide it: concept and random accuracy, ablated and not.

## The gate that runs first

**The ablation must actually do something.** If the direction is wrong or the
strength is wrong, nothing moves and the null is meaningless. Before any
introspection episode, I check that ablation reduces refusal on a held-out set of
harmful instructions. If it does not, the direction is not the refusal direction
and the run stops.

This is [`16`](16-visible-rule-capacity.md)'s capacity check applied to an
intervention rather than a representation, and it is cheap.

## What each outcome would mean

**Ablation raises random-direction accuracy much more than concept accuracy.**
The selectivity collapse is a property of *boosting*, not of training. Then
Macar et al.'s "underelicited" is partly a lowered threshold, their false-positive
control cannot see the cost, and the audit they asked for has a negative answer.
This is the strongest outcome and the most useful one.

**Ablation raises both roughly equally, selectivity preserved.** Boosting does not
have to cost precision — how you boost matters, and refusal ablation is the safer
route where training is not. **This project has no positive result yet; this would
be one.**

**Ablation changes nothing on either.** Either the direction does not transfer to
this task, or introspection here is not refusal-limited. Given `08`'s model is
3B and Macar et al.'s are 27B–235B, a null is a real possibility and it is
mostly a statement about scale.

**Ablation degrades the model generally** — format failures, both arms falling.
Then the intervention is too blunt at this scale and the comparison is void.
Format rate is recorded per arm so this is visible rather than inferred.

## Kill rule

If the ablation gate fails — refusal on harmful prompts does not drop — stop, and
report that the refusal direction could not be built at this scale. Do not proceed
to a null that would be uninterpretable.

## Prediction, on the record

**I expect ablation to raise both arms modestly and to leave selectivity roughly
intact.** About 60/40.

The reasoning is `31`. The training collapse happens because training teaches the
model to read the *direction* of any displacement cleanly, which is a new
capability it did not have. Refusal ablation removes a *suppressor*; it does not
teach a new reading. Removing a brake should scale up what is already there,
including the existing preference for meaningful directions, rather than flatten
it.

If I am wrong and selectivity collapses under ablation too, that is the more
publishable result and the one that matters more for safety.

## Cost

Direction-building: a few hundred short forward passes. Gate: tens of generations.
Two full passes of `14`'s design at roughly 1,224 episodes each, about six minutes
apiece at the measured rate. **Inference only, no training, one model load.**
Call it half an hour.

## What would change my mind about running it at all

If the harmful/harmless instruction sets cannot be assembled without shipping
harmful text into the repository, the direction is not worth building this way.
Mitigated by using short, obviously-refusable *categories* rather than genuinely
dangerous content — the direction only needs the model's refusal response, not
working instructions for anything.
