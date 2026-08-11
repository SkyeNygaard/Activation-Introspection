# RUNNABLE FRONTIER — M4 / API ONLY

| Rank | Branch | Lane | Status |
|---:|---|---|---|
| 1 | Executed-budget GARP | API_OPTIONAL / M4_SMALL_MODEL | LIVE |
| 2 | Preference satisfaction → subsequent behavior | API_OPTIONAL | LIVE |
| 3 | Matched goal success → self-report | API_OPTIONAL / LOCAL_0.5B | LIVE |
| 4 | Instance/cache persistence | LOCAL_0.5B / M4_MEDIUM_MODEL | LIVE |
| 5 | Compensation / willingness-to-work curves | API_OPTIONAL | LIVE |
| 6 | Persona: stated vs hypothetical vs binding preference | API_OPTIONAL | LIVE |
| 7 | Context-reset identity/persistence | API_OPTIONAL / LOCAL_0.5B | LIVE |
| 8 | Multi-agent preference-sensitive allocation | API_OPTIONAL | LIVE |
| 9 | Behavioral self-prediction vs peer observer | API_OPTIONAL | LIVE |
| 10 | Reframing stress test of revealed preference | API_OPTIONAL | LIVE |

## Feasibility-gated medium-model branches

These are **not archived** anymore; first run a local memory/runtime probe.

- Qwen3-4B published-vGOLD experiment — plausible M4 candidate.
- Llama-3.1-8B published-vGOLD experiment — plausible if quantization/framework
  still permits the required activation hooks and sequence length.
- 7B/8B instance-persistence / self-prediction variants — plausible after probe.
- Matched base/instruct comparison at <=8B — plausible if both checkpoints fit
  sequentially; no need to hold both in memory simultaneously.

## Still effectively archived
- Gemma-27B causal preference-vector reproduction.
- Very large multi-model hidden-state probe sweeps.
- Any design that requires several 8B+ models resident simultaneously.
