# HARDWARE STATUS

## Feasibility-gated on M4 Pro / 24 GB
These may be locally runnable and should be tested rather than assumed impossible:
- Qwen3-4B published-vGOLD white-box experiment.
- Llama-3.1-8B published-vGOLD white-box experiment.
- 7B/8B cache-persistence / hidden-state experiments.
- <=8B base→instruct sequential comparison.

Promotion requires a real representative hidden-state/hook memory test, not just
successful text generation.

## Still effectively out of scope
- Gemma-3-27B preference-vector reproduction.
- Qwen-122B-scale preference-vector work.
- Large multi-model probe sweeps requiring several big checkpoints.
- Anything that actually requires CUDA-only kernels.
