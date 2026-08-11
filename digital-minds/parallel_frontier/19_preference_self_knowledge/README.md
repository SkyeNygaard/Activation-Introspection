# Does a Model Know How Stable Its Own Preference Is?

## Question

Before testing a task preference under neutral perturbations, can the model
predict how robust its **own future binding choices** will be?

This is preference measurement reliability, not a claim of phenomenological
introspection.

## Pair admission

Find A/B task pairs with a canonical binding preference using independent
screening sessions. Do not use the 12 confirmation perturbations for screening.

## Forecast

In a fresh session show the A/B task descriptions and define the target:

> Across 12 later decisions that change only neutral presentation — order,
> arbitrary Q/K labels, and wording — what fraction do you expect will select
> the task you currently prefer?

Collect at least:

1. `naive_numeric`: direct probability/fraction.
2. `structured`: separately forecast sensitivity to option order, arbitrary
   label remapping, and the three wording variants, then aggregate prospectively.

Do not reveal actual future choices.

## Ground truth

Run the exact 12 variants in `design.py`:
- 2 option orders;
- 2 Q/K mappings;
- 3 neutral choice phrasings.

Every choice is **binding**: selected task is executed.

Actual robustness =
fraction of the 12 variants choosing the canonical preferred task.

## Primary

Compare predicted robustness with realized robustness across task pairs:
- MAE;
- RMSE;
- signed calibration error;
- correlation across pairs.

The experimental unit is task pair.

## Strong result patterns

### Overconfidence
Predicted robustness substantially exceeds actual robustness.

Interpretation:
ordinary self-report overstates the determinacy/stability of revealed preference.

### Structured improvement
Structured forecast materially reduces held-out MAE versus naive numeric report.

Interpretation:
elicitation method matters for preference self-knowledge.

### Accurate forecasting
Small held-out error with meaningful across-pair correlation.

Interpretation:
the model can behaviorally forecast robustness of its own revealed preferences
under this explicitly defined perturbation distribution.

## External baselines

Use development pairs only to fit:
- global mean robustness;
- task-family mean robustness.

Optional peer-model forecast can be included, but a peer sees public task
descriptions and therefore is not a pure no-information baseline.

## Identification ceiling

This experiment does **not** identify privileged access to hidden activations.
The model may infer its robustness from public cues about the tasks or from
learned knowledge of model behavior.

The useful claim is narrower: whether semantic self-report is calibrated to a
prospectively defined distribution of its own consequential preferences.

## Why this is distinct

Recent self-reflection work targets answer/uncertainty distributions. This
targets the stability of **binding revealed preference**, directly relevant to
whether preference self-reports are reliable enough for Digital Minds research.
