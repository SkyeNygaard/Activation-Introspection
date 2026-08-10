# Literature Boundary Map

This is a concise map of adjacent 2026 work discovered before branch construction.

## Sprint
Apart Digital Minds Research Sprint explicitly asks about:
preferences/trade-offs, welfare/valence, ground-truth introspection, multi-method elicitation,
persona/model identity, model-vs-instance-vs-conversation, and whether persona masks underlying signals.

## Preferences
- **AI Revealed Preferences** (Wang et al., 2026): consequential task choices; tedium aversion, leisure seeking, covert sycophancy; coherence/strength increase with capability.
- **Can Revealed Preferences Clarify LLM Alignment and Steering?** (Yamin et al., Microsoft, 2026): inferred cost functions from decisions; models have some coherence but weak faithful verbalization/steerability.
- **When Preferences Fail to Become Incentives** (Zhou & Ackerman, 2026): coherent reported utilities did not improve output quality when offered as incentives.
- **Probing Persona-Dependent Preferences** (Gilg et al., 2026): residual preference vector tracks/steers pairwise choice and is largely shared across personas.

## Welfare / affect
- **How's it going? Functional Welfare Axis** (Han, Chalmers, Izmailov, 2026): RL recruits pre-existing good/bad axis; causal effects on self-report, backtracking, refusal, uncertainty.
- **Whether, Not Which** (Keeman, 2026): mechanistic dissociation between affect reception and emotion categorization.

## Introspection
- **Quantitative Introspection** (Martorell, 2026): logit-based numeric self-report tracks/causally couples to emotive internal states.
- **Can LLMs Introspect? A Reality Check** (Singh, Linzen, Ravfogel, 2026): surface-cue/input-only controls undermine stronger introspection interpretations.
- **Masked by Consensus** (ACL 2026): self hidden-state correctness probes do not automatically outperform peer-model probes.
- **Introspection Adapters** (Anthropic, 2026) and **SAR** (Kutsyk & Zieliński, 2026): train adapters to report learned behaviors.

## Persona
- **Assistant Axis** (Lu et al./Anthropic, 2026): assistant persona lies along a cross-model activation axis; exists pretraining and can drift/steer.
- **Persona Steering on Capabilities** (Chen et al., 2026): persona interventions can change cognitive benchmark performance, not just wording.
- **Dynamic Persona Coherence** (ACL 2026): identity stability can be separated from adaptive psychological state in role-play systems.

## Implication
Avoid headlines already occupied:
- 'models have preferences';
- 'self-report tracks internal emotion';
- 'welfare steering changes self-report';
- 'persona changes behavior';
- 'a linear preference/persona vector exists'.

Prefer bridge questions and identification tests:
- goal-relative semantics,
- preference satisfaction → welfare,
- preference representation → incentive,
- state persistence → instance individuation,
- self-prediction vs observer,
- training changes causal coupling,
- causal axis interaction,
- multi-intervention measurement validity.
