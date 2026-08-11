# M4 24-GB MODEL FEASIBILITY GATE

Do not decide feasibility from parameter count alone.

For each candidate checkpoint, run the exact workload shape needed by the study.

## Candidate order
1. Qwen2.5-0.5B — confirmed baseline.
2. 1.5B / 3B family model.
3. Qwen3-4B.
4. Llama-3.1-8B or another 7B/8B checkpoint.

## Test
- load with intended framework;
- record dtype / quantization;
- tokenize representative longest prompt;
- forward pass;
- capture target-layer activation;
- apply one add/remove hook;
- run one fixed-choice scoring forward;
- if cache persistence matters, retain representative KV cache;
- record peak memory / swap / wall-clock.

## Promotion
PROMOTE if the exact scientific workload runs stably without pathological
swapping and is fast enough for the planned trial count.

## Important distinction
A model that can generate text locally may still fail the white-box experiment:
hidden-state capture, hooks, unquantized residuals, and KV cache can dominate the
weight-only memory estimate.
