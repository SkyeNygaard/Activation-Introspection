# Literature Boundary — What Would Actually Be New?

## 1. Functional-welfare axis

Han, Chalmers & Izmailov (2026) establish a reward-recruited functional-welfare
axis and show that causal steering changes self-report sentiment, backtracking,
confidence, and refusal. Adding vGOLD and subtracting vGOLD produce opposite
effects.

**Already occupied:** “steering a welfare-like direction changes semantic
self-report.”

## 2. Quantitative introspection

Martorell (2026) shows that logit-based 0–9 self-reports track probe-defined
emotive states, including wellbeing, and that activation steering causally shifts
those reports. Some measures become very strong at Llama-3.1-8B scale.

**Already occupied:** “numeric self-report can causally track an emotive internal
direction.”

The paper itself motivates skepticism from self-report/behavior dissociations.
A targeted search did not find the paper testing adversarial self-presentation,
persona masking, or an arbitrary opaque activation code.

## 3. Causal hidden-state codebook

The retained-trace repository's 2026-08-09 confirmation shows that
Qwen2.5-3B can learn an episode-specific opaque Q/K mapping from causally varied
hidden states with matched visible text. Target accuracy was 0.891; query-only
was exactly 0.5; target-query-only was +0.391.

**Already occupied by our own pre-sprint work:** “a model can learn an arbitrary
codebook from causally varied hidden states.”

## Remaining intersection

The plausible novel contribution is therefore the **intersection**, not any
ingredient:

> Does structured activation-grounded reporting of a causally imposed
> functional-welfare state remain accurate when ordinary semantic self-report is
> distorted by a persona/self-presentation intervention?

The experiment is valuable even without claiming novelty if it provides a clean
measurement-method extension requested by the sprint.

## Stronger Phase 2

If a recruited model passes, compare against a weak-recruitment model/vector.

This can separate:
- **generic access** to an injected activation;
- **semantic/functional welfare coupling** of that activation.

That distinction would materially sharpen what “reportable welfare direction”
means.

## Caution

Do not claim “first” from this search. Treat this as a targeted literature
boundary that must still be checked by teammates/mentors during the sprint.
