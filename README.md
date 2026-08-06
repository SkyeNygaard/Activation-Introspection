# A retained activation trace stays decodable after it stops being usable

Empirical AI-safety portfolio by Skye Nygaard, for SPAR Fall 2026. It holds one
executed study, two pilot repositories, and a claim ledger that lists the results
which did not survive.

The model is `Qwen2.5-0.5B-Instruct`. A concept vector is injected while a
neutral carrier's KV cache is built. The hook is then removed, and its absence is
asserted on every trial. Only after that is a freshly sampled concept-to-label
codebook revealed and the label scored. **The codebook cannot have been targeted
by the edit, because it did not exist yet.** Strength was frozen on a separate
development bank. The held-out bank was run once.

| inject layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| **use** (label accuracy, chance 0.125) | **0.500** | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| **storage** (probe on the same retained state) | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |

**The trace is fully retained. It stays linearly recoverable at readout depth from
every injection depth. What collapses is the model's ability to route it into a
symbolic lookup.**

## The obvious objection, and the control that answers it

A probe that recovers an injected direction may have recovered only what was
added. So I measured it.

I rebuilt the readout state as the *clean* carrier plus the identical delta, with
no forward computation in between. That scores **0.167**. The real arm scores
**1.000**. The delta on its own scores between 0.125 and 0.375. So the alignment
with the model's own natural-text representation is produced by the intervening
blocks, not by the edit.

There is one cell where the artifact really is present by construction: injection
site equal to readout site, with nothing computed in between. It comes out at
1.000 for both arms. That is what the artifact looks like when it is real.

![propagation control](activation-introspection/figures/retained_propagation.png)

## What it is, and what it is not

**It is a replication.** I ran a literature check against the design as built. The
transient-cache schedule turned out to be Lindsey's. The early-layer-only depth
profile is already published for Llama-3.1-8B. What is left as a contribution is
the answer space, meaning an arbitrary post-hoc codebook. That forecloses the
token-promotion artifact this repository previously fell for. The propagation
control above and the scale ladder are also mine. The boundary is drawn in
[LITERATURE-BOUNDARY.md](spar-application/LITERATURE-BOUNDARY.md).

The result licenses one thing: causal use of a retained trace, at early injection
sites, in this model, through this interface. It does **not** license
introspection, self-knowledge, or privileged access.

## The method, which is the point

The result above is a replication. The apparatus around it is not standard. I work
out what an experiment actually measures. I catch the artifacts that fake a result
in either direction. I drop a claim when a corrected comparison kills it.

[CLAIMS.md](spar-application/CLAIMS.md) grades every statement in this repository,
including the ones that did not survive. That list includes a retracted `r =
-0.774` headline which compared two different injection sites. It includes a "100%
identification" result killed by its own no-question control. It includes three
control arms that turned out to be arithmetic identities, so they could never have
failed. And it includes a randomized-response feedback channel that never
randomized.

I keep the same ledger in unrelated work. My [ARC White-Box Estimation
Challenge](https://github.com/SkyeNygaard/AI-Safety-Roadmap) repository carries a
`claims.csv` with per-claim evidence status, next to a graded competition
submission and a proof. Different field, same discipline. One evidence ledger is a
habit. Two independent ones is a method.

## Scope

This is **one executed study and two pilot repositories**. It is not six projects'
worth of results, and it does not pretend to be.
[PROJECT-BRIEFS.md](spar-application/PROJECT-BRIEFS.md) records what I would do on
each of the six SPAR projects I am applying to. It is deliberately unflattering
about the three that are proposal-only.

## Where to go next

- [`spar-application/`](spar-application/) is **the full write-up.** It covers the
  gates that passed and the gates I withdrew, states the damage matching honestly,
  gives the scale ladder, and maps this work onto each of the six projects.
- [`activation-introspection/`](activation-introspection/) holds the retained-trace
  study and its apparatus.
- [`adaptive-monitor-sandbox/`](adaptive-monitor-sandbox/) holds the control
  sandbox and the Study 3 design.
- [`AI_ASSISTANCE.md`](AI_ASSISTANCE.md) records what was agent-assisted. Most of
  it was, including the review pass that found several of the defects listed
  above. That is a useful check. It is not an independent one.

## Reproducing it

Each subdirectory carries its own `pyproject.toml`, tests, and verification
contract. Both pass from a fresh clone:

```bash
cd activation-introspection && make setup && make check
cd ../adaptive-monitor-sandbox && make setup && make check
```

Every headline number can be regenerated from committed raw per-trial rows. Config,
prompt hashes, model revision, and environment are recorded alongside them.
[AUDIT-MANIFEST.md](spar-application/AUDIT-MANIFEST.md) gives the commands and the
artifact policy.

The two code repositories were built separately and merged here with `git
subtree`, so their individual commit histories are intact. That history records
the corrections as they happened, and it is part of the evidence.

## License

MIT. See [LICENSE](LICENSE).
