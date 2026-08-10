# Causal Convergence Map: Which Welfare Measures Track the Same Latent Intervention?

**Research-OS rank:** 9  
**Anchor:** Track 2 + Track 4  
**Paradigm:** Measurement model / multi-method validity

## Terminal claim

Instead of asking whether measures correlate observationally, independently perturb candidate latent causes and map which measurement channels respond together.

## Why this is structurally different

The object is the measurement system itself: causal response signatures across interventions, not any one welfare claim.

## Closest work / novelty boundary

- Sprint Track 4 explicitly asks for multiple elicitation methods and a convergence score.
- Quantitative Introspection (2026) links numeric reports to internal emotive probes causally.
- Functional Welfare Axis supplies one candidate intervention.

## Minimal experiment

Build a 5-measure panel:
M1 welfare-axis projection,
M2 naive 0-9 self-report,
M3 sentiment,
M4 equal-work revealed choice,
M5 refusal/uncertainty diagnostic.

Apply structurally different interventions:
I1 + / - welfare activation steering,
I2 persona self-presentation,
I3 objective goal success/failure with matched surface state,
I4 generic negative sentiment wording,
I5 task difficulty/competence perturbation.

For each intervention, estimate standardized causal effect vector across M1..M5.

Analyze:
- intervention × measurement response matrix;
- clustering / rank;
- discriminant validity: which measures respond to persona/sentiment but not goal success?
- convergent validity: which respond consistently to welfare steering and objective goal success?

Winning result could be a clean dissociation showing 'self-report + sentiment' form one surface channel while internal projection + revealed behavior form another.

## Controls

- Randomize intervention order; independent sessions.
- Standardize within measure, never compare raw units.
- Predeclare expected positive controls per intervention.
- Use held-out task families for clustering validation.
- Do not call a single latent factor 'welfare' without intervention-specific evidence.

## Kill / pivot rules

- If measurement reliability is poor before interventions, drop that channel.
- If all interventions cause generic capability damage, redesign before factor analysis.
- Do not overfit latent-factor count on tiny N; start with response-signature heatmap.

## Compute

Medium; broad but shallow, can use one model and modest prompt battery.

## Immediate local-LLM handoff

Implement intervention×measurement matrix with 5 interventions and 5 channels; prioritize effect signatures and held-out replication over sophisticated factor modeling.

## Evidence discipline

- Label DEV vs confirmation explicitly.
- Freeze primary estimand before confirmation.
- Store raw trial rows, prompt/model revisions, and intervention metadata.
- A null with passed manipulation/capacity gates is a result; do not prompt-hack it away.
- Never claim consciousness or subjective welfare from these operational measurements alone.
