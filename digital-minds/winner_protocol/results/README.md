# Results — first real model evidence in this branch

Everything here is Qwen3-4B-Instruct-2507 on local MPS, public
`davidafrica/functional-wellbeing` vector `qwen3-4b_step400/goal/mean_diff.pt`,
weights and vector cached under `activation-introspection/hf_cache`. Before this,
every number in this package was simulated. See DECISION_LEDGER R16–R18.

Run anything here with:

```bash
HF_HOME=../../activation-introspection/hf_cache HF_HUB_OFFLINE=1 python <script>
```

and run `python -m introspect.preflight qwen3-4b` from `activation-introspection`
first — a 4B run needs ~10.7 GiB free on this machine.

| Artifact | What it establishes |
|---|---|
| `naive_semantic_qwen3-4b_L29_a2.jsonl` | Step 0 at R10's L29: state effect −0.051 / −0.003 / +0.007, persona effect **+6.13**. Readout live, hidden state does nothing. |
| `debug_source_effect_L29.jsonl` | Scope × factor at L29. Marker-only is noise at every factor; all-positions reaches only +0.42 at factor 32, on top of general degradation. |
| `localize_source_layer.jsonl` | All 36 layers, source convention. Unimodal peak at **L22** (+3.96 at factor 2), dead from L25. R10 wrong, R8 right. |
| `debug_source_effect_L22.jsonl` | At the good layer, the marker edit is still inert: −0.054 at factor 32, a 4×-residual edit. |
| `visible_dev_qwen3-4b.jsonl` | Capability control: **48/48**, mass and format 1.000. The task is not the problem. |
| `naive_semantic_qwen3-4b_L22_a2.jsonl` | Root gate 1 at L22 under the protocol's own marker convention: +0.0156, i.e. noise that still passes a `> 0` gate. |
| `dev_qwen3-4b_L22_a2_neutral.jsonl` + `_summary.json` | The DEV cell. `target − query_only` = **0.000056** against a 0.10 gate; target accuracy exactly 0.500; mass and format 1.000. |

## Second pass — the carrier repair (R19–R20)

| Artifact | What it establishes |
|---|---|
| `span_threshold_L22.jsonl` | How much steered span the payload needs. 1 token +0.078, 5 tokens +0.301, **51 distal tokens +1.651**, 104 tokens +3.961. The state propagates; mass and proximity both matter. |
| `dev_qwen3-4b_L22_a2_neutral_span49.jsonl` + `_summary.json` | Repaired cell, each state widened to the 49 tokens ending on its marker. `target − query_only` = **−0.0058**, accuracy **0.500**, mass and format 1.000. |
| `answer_site_probe_L22_w49_perm.json` | Query sign decodable from the final-layer residual at the answer position: **0.875** leave-one-out, permutation **p = 0.0005**. Per carrier 0.833 / 0.625. |
| `answer_site_probe_L22_w49_residue_control.json` | The decode is not our own injection. With `vGOLD` projected out: **0.854**, p = 0.0005; decoder direction's cosine with `vGOLD` only **0.100**. |

**Verdict.** The first pass was an instrument failure. The second pass is a
**scoped null**: with the carrier repaired, the state is demonstrably present at
the readout site (p = 0.0005), the model binds arbitrary labels perfectly when
that state is visible (48/48), the output stays well formed, and the report of
the injected state sits at exactly chance.

Two cautions before this is quoted. The same-context semantic channel reads
**8.99 out of 9 at every span width** — it is at ceiling and cannot serve as a
manipulation check; use the naive channel. And the closure is narrow: one model,
layer, vector, factor, persona, two carriers, and a *linear* decoder.
