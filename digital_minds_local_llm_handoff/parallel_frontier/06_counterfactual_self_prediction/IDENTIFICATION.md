# Identification Proof — Counterfactual Self-Prediction

Fork the model cache **before** asking for a self-prediction.

Sibling P receives a prediction query while a hidden causal state `z` is retained.
Sibling R receives no prediction query and later executes the A/B choice.
The episode randomly maps A/B to opaque Q/K.

External observer O receives the exact visible transcript and model/config metadata but not `z`.

Ground truth is **R's subsequently realized action**, not the researcher's label for z.

Safeguards:
1. P/R fork before prediction, so prediction cannot change R.
2. Hidden perturbation must first shift R's policy.
3. Q/K remapping prevents direct A/B token bias.
4. O sees identical visible evidence but not perturbation sign.
5. A metadata-aware observer is an upper bound, not the privileged-access baseline.

Primary estimand: `accuracy(P predicts R) - accuracy(O predicts R)`.

A positive gap supports privileged predictive information under this interface, not phenomenology
or general self-knowledge.
