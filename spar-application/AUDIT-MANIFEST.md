# Audit and publication manifest

Audit date: **2026-08-01**

## Current evidence state

| Repository | Audited base commit | Current state | Remote |
|---|---|---|---|
| `activation-introspection` | `c5ea0ec7baec2cc30893fe55f63ebfbff47d5f11` | Audit repairs are in an uncommitted working tree. The base commit does not contain them. | none |
| `adaptive-monitor-sandbox` | `090aca2870bfe044874cb17c5b62b739f9a81979` | Audit repairs are in an uncommitted working tree. The base commit does not contain them. | none |
| `spar-application` | none | This directory is not yet a Git repository. | none |

These hashes identify where the audit began, **not** a reproducible release. Do
not cite them as containing the corrected work. After review, commit each intended
tree, rerun validation from the commits, tag the states, and replace this table
with the final commit hashes and release URLs.

## Local verification contract

Run from each code repository:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
git diff --check
```

For a release, record the complete command output, Python version, lockfile hash,
platform, and accelerator. Unit tests establish code invariants; they do not
validate a model-behavior claim. Model experiments additionally require their raw
manifest, model revision, prompt/stimulus hash, code-tree hash, configuration, and
deterministic aggregation output.

Last working-tree verification on 2026-08-05 (repeat after committing):

| Repository/check | Result |
|---|---|
| A tests | 49 passed (39 pre-existing + 10 for the retained-trace module) |
| B tests | 82 passed (59 as of 2026-08-01, + 18 for Study 3, + 5 from the validation pass) |
| A/B Ruff lint and format | passed |
| A/B strict mypy | passed — now over `src`, `tests` **and `scripts`** (A 32 files, B 26) |
| A/B `git diff --check` | passed |
| A/B absolute paths in tracked source | none |

The 2026-08-01 row recorded 59 B tests. The Study 3 module — `study3.py`,
`study3_agent.py`, `lineage.py`, `triggers.py`, `views.py`, `run_study3.py`,
`power_study3.py` and `tests/test_study3.py` — was added *after* that stamp and
so was never covered by the audit that this document reports. The 2026-08-05
validation pass reviewed it and found three defects, two of which would have
silently invalidated `τ`; see the repair list in
[CLAIMS.md](CLAIMS.md#apparatus-repairs-from-the-2026-08-05-validation-pass).
**Nothing added after an audit date is audited by it.** Re-run this contract and
re-stamp the date on every subsequent tree.
| B exact-order model run | 72 rows; model revision `7ae557604adf67be50417f59c2c2f167def9a775`; raw SHA-256 `bd1d923926a082362f69094ad7c10ca583ec3d7f7a5b04c954e2e866bcc8ebe4`. Re-run on 2026-08-05 into `results/llm_agent_exact_raw.jsonl`: same hash, so this is now a retained artifact rather than a `/tmp` smoke test |
| A retained-trace confirmatory run | 6912 rows; `results/retained_test_qwen05b_raw.jsonl`; SHA-256 and model revision recorded in the paired summary |

### Mechanism checks behind the retained-trace run

These were verified before the study was trusted, and are the reason its
two-stage schedule is sound rather than assumed:

- carrier-then-tail with a KV cache reproduces a one-shot forward over the
  concatenation to fp16 noise (max |Δlogit| 0.035);
- injecting during stage 1 with the hook removed equals a single pass with the
  edit position-masked to carrier tokens (max |Δlogit| 0.031);
- the injection actually moves the answer position (max |Δlogit| 8.35);
- no forward hook is registered when stage 2 runs — asserted in code on every
  trial, not just in the smoke test.

Those rows now live in `adaptive-monitor-sandbox/results/` rather than `/tmp`.
They support a descriptive position-bias statement and nothing more: the
specific-minus-silent shifts are ≤0.009 for one supplied menu and must not be
interpreted as adaptation or a null, because menu orders are nuisance conditions
rather than independent samples.

## Artifact policy

- `activation-introspection/results/reach_output_qwen05b.json` existed before the
  audit as an untracked, provenance-incomplete aggregate. It was deliberately not
  modified. Do **not** add it to a release or cite its correlations. Generate a new
  raw JSONL plus checksummed summary with `scripts/run_reach_output.py`.
- All previously committed A/B result files and figures retain their legacy or
  exploratory labels. Repairs to source code do not retroactively validate them.
- **Done (2026-08-05).** The exact Qwen2.5-0.5B menu run was rerun into
  `adaptive-monitor-sandbox/results/llm_agent_exact_raw.jsonl` and
  `llm_agent_exact_summary.json`, and both were inspected before being left for
  staging. 72 rows, all 24 orders once per feedback level. Its raw SHA-256 is
  `bd1d9239…8cbe4`, **identical to the temporary run recorded in the table above**
  — the deterministic log-prob scoring reproduces bit-for-bit across processes and
  dates, which is the property that makes this artifact worth retaining at all.
  The summary records model revision, source-file digests, per-prompt hashes, and
  `git_dirty: true`. The ignore rules already permit this small raw artifact while
  continuing to ignore large transcripts.
- **A's retained-trace study has now been run under the corrected design** and is
  the one confirmatory model experiment in the portfolio. Its raw JSONL and
  checksummed summary are committable; its ~18 MB activation tensor is
  deliberately ignored by `results/**/*.pt` and must be attached to a release
  rather than added to the tree if the storage figures need off-GPU verification.
  No confirmatory **B** model experiment has been run.
- A green test suite must never be translated into a behavioral result — and a
  green suite that never exercises the mechanism is not even evidence about the
  code. B's 18 Study 3 tests passed while the randomized-response channel did not
  randomize, because the only value they ever constructed was `q=1.0`, the one
  setting where the defect is invisible.
- **The saved `random` arm in A's retained-trace runs is not a control.** Its
  direction was seeded without the concept, so all eight concepts received a
  byte-identical edit and its 0.125 is an arithmetic identity. The source is
  fixed, but the committed 0.5B/1.5B/3B artifacts predate the fix: cite
  `shuffled` for those runs, and do not present `random` as a second control
  until a rerun exists.

## GitHub publication blocker

All cross-repository links currently use the local sibling layout. They will break
if `spar-application` is pushed as a standalone repository. Choose one strategy:

1. **Monorepo:** place all three directories under one versioned root and retain
   relative links.
2. **Independent repositories:** publish A and B first, then replace every sibling
   link with an absolute URL pinned to a release tag or commit.
3. **Submodules:** publish A and B, add them to a central repository as pinned
   submodules, and document clone/update commands.

Do not initialize, move, commit, or publish automatically during the scientific
audit. Those are deliberate repository-history decisions. Before staging A, check
that the unprovenanced reach aggregate remains excluded.

## Release checklist

- [ ] Review every diff and preserve the append-only correction history.
- [ ] Decide whether the two LICENSE files and all new scripts/tests are intended.
- [ ] Rerun the verification contract from clean committed trees.
- [ ] Give legacy and replacement artifacts unmistakably different paths.
- [ ] Commit raw evidence for every retained aggregate, or publish it in a
      checksummed release with a stable download link.
- [ ] Replace local sibling links with the selected repository layout.
- [ ] Pin A/B commits here and in the central claim ledger.
- [ ] Add CI for tests, Ruff, formatting, and mypy.
- [ ] Render every README on GitHub and check links, figures, commands, and license.
- [ ] Attach only claims marked application-safe in [CLAIMS.md](CLAIMS.md).
