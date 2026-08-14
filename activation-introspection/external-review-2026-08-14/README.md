# External review, 2026-08-13 / 2026-08-14

Pulled into the repository on **2026-08-14** from a Google Drive folder where it
had exactly one copy. Nothing here was produced on this machine.

## What this is

An outside re-analysis of this project's saved raw artifacts, run in a sandbox
that had the result files but **no model weights, no `transformers`, no `peft`,
and no saved LoRA adapters**. It could therefore re-score rows that already
existed; it could not run the model. Everything in `analysis/` is a **post-hoc
secondary analysis of artifacts frozen for other purposes**, and the reports say
so themselves.

Read that limit literally. Nothing here is a new measurement. The strongest new
thing in it — the latent-XOR quartet — is a stricter way of scoring rows this
repository already had, discovered after seeing the data. It needs a prospective
run before it can be quoted as a result.

## What it changed here

Three descriptions in this repository were wrong, and are corrected in place. The
corrections and the evidence for them are in [`HANDOFF.md`](../../HANDOFF.md) §0.
In short: the `random` arm is not a no-signal arm; the structural null on twin
pairs is 0 rather than 0.25; and notes/14 is not evidence of semantic abstraction.

The review's own account of what it found is in
`reports/Activation-Introspection-Sandbox-Audit-2026-08-13.md`. Its claims about
this repository were checked against the code before anything was rewritten — the
`random`-arm claim against
[`run_remap_training.py`](../scripts/run_remap_training.py) lines 493–503, the
null against `results/README.md`, which had recorded the correct value all along.

## Layout

| path | what |
|---|---|
| `reports/00_HANDOFF_*.md` | the review's own handoff; read first |
| `reports/Activation-Introspection-Sandbox-Audit-*.md` | the audit of this repo, its claims and its literature boundary |
| `reports/Sandbox-Continuation-*.md` | latest consolidated analysis and the frontier it proposes |
| `reports/Sandbox-Deep-Pass-*.md`, `reports/Sandbox-Followup-*.md` | supporting passes |
| `analysis/*.py` | the analyzers, so every number here can be re-derived |
| `analysis/*.json` | their machine-readable outputs |
| `analysis/next_latent_binding_protocol_draft_v1.json` | proposed next protocol — **draft, not frozen, not run** |
| `Sandbox-Continuation-Bundle-2026-08-13.zip` + `Sandbox-Continuation-SHA256SUMS.txt` | the checksummed package |

All 16 checksums in `Sandbox-Continuation-SHA256SUMS.txt` verified on import:

```bash
cd activation-introspection/external-review-2026-08-14/analysis && shasum -a 256 -c ../Sandbox-Continuation-SHA256SUMS.txt
```

## Not imported

- `Archive(2).zip` (~30 MB) — a snapshot of this repository's own top-level
  documents and `papers/` cache as of 2026-08-12/13. Already in git history; not
  worth a 30 MB duplicate. It remains in the Drive folder.
- `Archive(3).zip` (~308 MB) — a raw snapshot the review could not transfer
  through the Drive connector. It was never in the Drive folder. The live repo is
  the canonical source; request that archive separately only if a raw snapshot is
  specifically needed.

## Standing warnings from the review

1. Do not reintroduce random-as-noise or false-positive language.
2. Do not use 0.25 or 0.5 as the structural twin/quartet null.
3. Treat latent-XOR as exploratory until prospectively rerun.
4. The held-out semantic dataset has few genuinely independent semantic cases
   despite many rows; do not generalize the failure beyond what was tested.
5. Do not continue the notes/29–37 prompt branch with another descendant.
