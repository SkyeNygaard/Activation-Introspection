# HARD COMPUTE SCOPE — M4 MACBOOK PRO

This overrides older execution recommendations.

## Available
- M4 MacBook Pro.
- Confirmed cached Qwen2.5-0.5B.
- CPU/MPS-small-model experiments.
- Model-free statistics/simulation.
- Behavioral/API experiments if ordinary API access is available.

## Hardware
- M4 MacBook Pro with **24 GB unified memory**.
- No CUDA/NVIDIA GPU.
- Apple MPS / MLX / llama.cpp style local inference may make substantially larger
  quantized models feasible than a CPU-only framing would suggest.

## Out of scope
- Any branch whose minimum viable version *requires* rented CUDA hardware.
- 27B+ white-box activation studies as a default.
- Large multi-model hidden-state probe sweeps.
- 'Run it on Modal' as the default next action.

## Execution lanes

**LOCAL_0.5B**
- confirmed cached Qwen2.5-0.5B;
- ideal for fast protocol/debug/cache experiments.

**M4_MEDIUM_MODEL**
- candidate 3B–8B models;
- must pass a real local feasibility gate before becoming LIVE;
- quantized inference may fit comfortably, but white-box PyTorch/Transformers
  activation work can require much more memory than weight-only inference.

Feasibility gate:
1. load model locally;
2. run one representative prompt;
3. capture the exact hidden layer needed;
4. install/remove one intervention hook;
5. run the intended sequence length;
6. observe memory pressure / swapping;
7. promote only if interactive runtime remains practical.

**API_OPTIONAL**
- consequential behavioral protocols with no white-box dependency.
