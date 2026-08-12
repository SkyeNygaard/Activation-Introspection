# Natural-state transfer pilot: stopped at the reachability gate

Run date: **2026-08-10**

## Question

Can the existing episode-remapped `Q/K` interface classify a state that
`Qwen2.5-3B-Instruct` produced while following a two-hop route, rather than a
planted concept direction?

This was an inference-only feasibility pilot. Five matched route worlds used the
same two bridge nodes (`maple` and `cedar`) and different endpoints. Twin donor
prompts differed only in the selected first edge. The planned report used four
donor worlds as demonstrations and the fifth as the query, with byte-identical
visible report text and the existing exact 24-cell order × mapping × query design.

The report was gated on a more basic requirement: replacing the residual state at
the route marker had to make the ordinary route answer follow the donor state.
Model revision `aa8e725…04d1` and layer 9 were fixed before the run.

## Two implementation versions

Smoke protocol V1 produced no raw result. The existing
`Intervention(mode="replace")` normalizes its direction on the model device. With
a float32 captured donor and a bfloat16 recipient, that did not reproduce the
donor tensor closely enough and the runner stopped on its exact-replacement
assertion.

V2 changed only this plumbing. A dedicated hook assigns the captured residual
directly, checks the recipient state before editing, checks the state after
editing, and always removes itself. A self-patch reproduced the full next-token
logit vector exactly: maximum absolute error **0.0**.

## V2 result

The reachability gate failed, so **no `Q/K` reporting rows were run**.

| gate | frozen requirement | result |
|---|---:|---:|
| clean unrestricted route answers | 10/10 | **8/10** |
| exact self-patch | max logit error ≤ 1e-4 | **0.0** |
| bidirectional cross-patch | ≥4/5 worlds | **0/5** |

The prompt itself was not fully reliable: the positive route lost to the negative
endpoint in two worlds. More importantly, even in cleanly solved worlds, swapping
the layer-9 marker state did not make the final answer follow the donor in both
directions. The visible route tokens remained available to downstream attention,
so this marker/site was not a causal bottleneck for the answer.

The saved V2 summary also contains a huge negative “normalized recovery” value.
Do not cite it. When a clean donor margin was negative, the pilot divided by a
`1e-6` floor. The independent clean-answer and bidirectional-cross-patch gates
already failed, so this diagnostic bug did not change the stop decision. The
executed source hash remains in the V2 protocol; the current runner now uses
`(patched - recipient clean) / (donor clean - recipient clean)` and requires a
new protocol rather than rewriting V2.

## What follows

**Observation:** exact state replacement works, but the tested route-marker state
at layer 9 did not control the ordinary answer.

**Interpretation:** this is an instrument failure, not a null result on natural-
state reporting. The reporter was never evaluated, so the pilot does not show
that the existing introspective capability is injection-specific.

**Closed in scope:** this exact route prompt, marker position, layer, and
single-position transplant should not be repeated or rescued with report-prompt
tuning.

**Reopen condition:** on a disjoint development bank, first localize a token/layer
whose cross-patch changes the ordinary answer under a reliable clean task. Freeze
that site and a fresh query bank before running the `Q/K` reporter. An output-ready
answer state would be easier to reach but supports a narrower claim than a genuine
intermediate, and should be labeled accordingly.

**That successor ran on 2026-08-11.** Its frozen protocol also stopped at
reachability — the output-ready version fixed the clean-task defect, 10/10 at
probability 1.000, and still controlled the answer in 0/5 tasks at layers 9, 21
and 26 — but a post-hoc all-layer localization on the development bank then
satisfied the condition stated above. From **block 27** the same single-position
transplant makes the ordinary answer follow the donor in 10/10 transplants; below
it the state does not carry the answer at all. The layer-9 site tested here was
not merely a weak choice, it was well below where the answer exists.

A third protocol then froze block 27 and required it to pass the same gate on a
never-scored held-out bank. It did not — 5/5 on development, 3/5 held out — so
the reporter has still not run. See
[`10-output-ready-arithmetic.md`](10-output-ready-arithmetic.md).

## Artifacts

- V1 stopped protocol: `results/natural_state_smoke_protocol_v1.json`;
- V2 protocol, raw rows, manifest, and summary:
  `results/natural_state_smoke_protocol_v2.json`,
  `results/natural_state_smoke_v2_raw.jsonl`,
  `results/natural_state_smoke_v2_raw.manifest.json`, and
  `results/natural_state_smoke_v2_summary.json`;
- runner and exact patch implementation: `scripts/run_natural_state.py` and
  `src/introspect/natural_state.py`.
