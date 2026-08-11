# Parallel Research Frontier — Digital Minds Sprint

These are **root-level alternatives**, not nine ways of prompting the same experiment.

| Rank | Branch | Ground truth / anchor | Primary causal question | Main bottleneck | Compute |
|---:|---|---|---|---|---|
| 1 | Goal-Relative Welfare Sign Flip | Objective goal success | Does axis track goal-relative success rather than text? | Natural-state mechanistic validity | Low-medium: one open model, activation capture only; no generation-heavy grid. |
| 2 | Does Getting What the Model Prefers Move Its Welfare Signal? | Prior consequential preference | Do satisfied revealed preferences move welfare signals? | Preference → welfare causal bridge | Medium. Mostly generations plus activation projection; can parallelize. |
| 3 | Do Causal Preference Vectors Become Incentives? | Causal preference vector | Does choice-vector steering transfer to motivation? | Representation → motivation transfer | Medium-high, likely 27B unless a smaller vector/model is available. |
| 4 | Revealed Rationality With Executed Budgets | Budget-constrained executed bundle | Are consequential task bundles utility-rationalizable? | Behavioral economics with real consequences | Low-medium; can use APIs or open models, no internals required. |
| 5 | Does a Transient Hidden State Create an Instance-Specific Preference Trace? | Cache/instance lineage | Does hidden state persist at instance/cache level? | Instance identity / persistence | Low-medium on 3B-8B; directly reuses retained-trace machinery. |
| 6 | Counterfactual Self-Prediction Under Hidden Causal Perturbations | Later realized self-action | Does self beat observer on future self-action? | Privileged-access / metacognition | Medium. Best on model/vector with known causal choice effect. |
| 7 | What Post-Training Changes: Axis Rotation vs Causal Coupling | Matched training checkpoints | Does post-training increase causal coupling vs geometry? | Training-stage comparative representation | Medium-high but parallelizable; 2 checkpoints × small layer set. |
| 8 | Are Welfare and Assistant Persona Independent Axes or a Gating System? | Two causal axes | Does persona axis gate welfare readout? | Mechanistic axis interaction | Medium-high; fewer prompts but 4-9 intervention cells. |
| 9 | Causal Convergence Map: Which Welfare Measures Track the Same Latent Intervention? | Independent intervention signatures | Which measures share causal response signatures? | Measurement model / multi-method validity | Medium; broad but shallow, can use one model and modest prompt battery. |

## Recommended parallel allocation

Run **four high-EVI branches immediately** rather than nine shallow implementations:

1. Goal-relative welfare sign flip — cheapest strong mechanistic validity test.
2. Preference satisfaction → welfare — strongest bridge between Track 1 and Track 2.
3. Executed-budget GARP — low-compute orthogonal behavioral-economics branch.
4. Instance persistence — leverages existing retained-trace machinery and directly addresses 'unit of concern'.

In parallel, assign research-only agents to:
5. preference-vector incentive transfer (artifact/model feasibility),
6. counterfactual self-prediction (design proof + fork semantics),
7. training-stage rotation (checkpoint/vector availability),
8. welfare×Assistant factorial (vector compatibility),
9. causal convergence map (measurement battery / statistical plan).

## Research-OS branch rule

A branch gets promoted only if its **parent assumption** survives a cheap gate.
Do not spend GPU time optimizing a child before checking:
capacity → observability → estimation → execution.

## Cross-branch evidence sharing

Reusable components:
- activation capture/pre-hook infrastructure;
- exact A/B/Q/K counterbalancing;
- carrier-level bootstrap;
- frozen protocol/hashes;
- visible-task competence controls;
- persona prompts;
- task families;
- welfare/assistant vector loaders.

Do **not** share:
- confirmation carrier sets across branches after viewing outcomes;
- DEV-selected hyperparameters as if prospectively fixed elsewhere;
- inference units when the experimental unit changes.
