# SANDBOX PROGRESS BATCH 2

## Structural discoveries

1. **Old GARP semantics were wrong at the experiment level.**
   Work performed is a bad; algebraically correct GARP code did not justify
   monotone-consumer interpretation.

2. **Branch 04 V3** now uses work *skips* and makes finite-menu revealed cycles
   primary. Model-free power check:
   - stable log-utility agents: 0/99 false cycle failures;
   - random Pareto choices: ~99.7% cycle rejection.

3. **Branch 12 was pruned.**
   Existing 2026 work already studies whether preferences predict downstream
   behavior.

4. **Branch 18 replaces it** with preference path dependence:
   randomized recent A/B workload -> next binding preference, crossed with
   full-history / summary-only / blank-reset.

5. **Branch 19 added:** preference self-knowledge.
   Forecast robustness across 12 fixed neutral binding-choice perturbations,
   then compare forecast with realized robustness.

6. **M4 feasibility is executable.**
   `m4_feasibility/m4_whitebox_probe.py` tests direct MPS load, target-layer
   capture, pre-hook editing, logits and optional KV cache with allocator stats.

7. **Shared binding tasks added.**
   Six deterministic, auto-gradable microtask families and counterbalanced Q/K
   choice prompts prevent hypothetical-choice leakage.

## Validation

`validate_research_os_frontier.py` currently passes 12 model-free checks:
syntax, task binding, goal counterbalance, Branch 04 falsification, Branch 18
synthetic recovery, Branch 19 synthetic calibration, winner digit fix, and
synthetic-fixture quarantine.

## Immediate handoff

Run `LOCAL_AGENT_BATCH_2.md`.
