# Audit and publication manifest

Audit date: **2026-08-10**

## Current evidence state

| Repository | State | Layout |
|---|---|---|
| `activation-introspection` | Retained-trace history, the 2026-08-09 causal-codebook V1 precursor and V2 repair-confirmation, and the 2026-08-10 Stage 1a/1b attention screens. Stage 1b returned a pre-registered stop. The new worktree artifacts are checksummed but not yet committed. | merged at `activation-introspection/` |
| `adaptive-monitor-sandbox` | Repairs, the Study 3 module, and the retained exact-order artifact committed on `main` (`457f487`, on top of `090aca2`). | merged at `adaptive-monitor-sandbox/` |
| `spar-application` | Committed. | `spar-application/` |

All three now live in one repository. The two code repositories were merged with
`git subtree`, so their individual histories are preserved rather than flattened.
The commit sequence is itself part of the record. Independent repositories
would have required every cross-document link to be rewritten to a commit-pinned
URL and re-pinned on each push; an audit trail a reader cannot follow is worth
nothing.

No remote is configured. Pushing is a separate, deliberate decision about what
becomes public. Before pushing, re-run the verification contract from a clean
clone and confirm the unprovenanced reach aggregate is still excluded.

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

Last working-tree verification on 2026-08-09 (repeat after committing):

| Repository/check | Result |
|---|---|
| A tests | 74 passed, including causal-codebook, attention-patching, and fail-closed artifact analysis |
| B tests | 82 passed (59 as of 2026-08-01, + 18 for Study 3, + 5 from the validation pass) |
| A/B Ruff lint and format | passed |
| A/B strict mypy | passed, now over `src`, `tests` **and `scripts`** (A 48 files, B 26) |
| A/B `git diff --check` | passed |
| A/B absolute paths in tracked source | none |

The 2026-08-01 row recorded 59 B tests. The Study 3 module (`study3.py`,
`study3_agent.py`, `lineage.py`, `triggers.py`, `views.py`, `run_study3.py`,
`power_study3.py` and `tests/test_study3.py`) was added *after* that stamp and
so was never covered by the audit that this document reports. The 2026-08-05
validation pass reviewed it and found three defects, two of which would have
silently invalidated `τ`; see the repair list in
[CLAIMS.md](CLAIMS.md#apparatus-repairs-from-the-2026-08-05-validation-pass).
**Nothing added after an audit date is audited by it.** Re-run this contract and
re-stamp the date on every subsequent tree.

### Retained model artifacts

| Artifact | Verification |
|---|---|
| B exact-order model run | 72 rows; model revision `7ae557604adf67be50417f59c2c2f167def9a775`; raw SHA-256 `bd1d923926a082362f69094ad7c10ca583ec3d7f7a5b04c954e2e866bcc8ebe4`. Re-run on 2026-08-05 into `results/llm_agent_exact_raw.jsonl`: same hash, so this is now a retained artifact rather than a `/tmp` smoke test |
| A retained-trace confirmatory run | 6912 rows; `results/retained_test_qwen05b_raw.jsonl`; SHA-256 and model revision recorded in the paired summary |
| A causal-codebook V2 repair-confirmation | 576 episode rows / 2,880 scored forwards; model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`; raw SHA-256 `f45d2ac5…7cf20`; manifest `84022779…fc40`; summary `896fcdd6…37f6a`; protocol `fbba4892…ffc39`; config `06e404fa…61f28`; analyzer `67130c64…94f57`; exact 8 × 3 × 24 design and all 40 statistics independently reconstructed with no discrepancy |
| A DEV attention-localization screen | One concept/carrier, 12 rows, 1,248 patches / 1,284 scored forwards; raw `530f4f55…d5c1c`; manifest `8c6dffb8…a5e0c`; summary `f2329275…d3dcbb`; protocol `27c8af5f…e41427`; analyzer `025a0add…6d2c`; independently reconstructed with no discrepancy. Selection only, not confirmation |
| A trained zero-demonstration reporter | 504 rows per version; V2 raw `a3d6361e…db6c68`, protocol `6812dc8f…e461d`; V1 precursor raw `13641578…f796e4`, protocol `acb92e81…d8082`; runner `21307f54…20cbc`; analyzer `a82e0e00…b20e9`. V2 passes all four gates at 0.583 twin-pair on held-out directions; V1 fails the verbalization gate under the current analyzer. One training seed. Regenerate with `make report-training-report` |
| A Stage 1b DEV head screen | Three concepts × two carriers, 72 unit rows, 5,112 scored forwards; raw `9833a9bf…54bde`; config `d29e0b5c…f0c6`; protocol `759c0850…25856d`; runner `93c8d4e2…41a0`; analyzer `de69e02d…63078`; summary `c9bea45f…24212`. The analyzer evaluates the frozen stop/go predicates itself and returned `proceed=false`. Protocols V1 and V2 are retained unrun as disclosed precursors. Regenerate with `make head-screen-report` |
| A programmatic-attention lowering | 818 raw rows; raw `c62c1f77…ba1a7b`; protocol `7ad77347…96f593`; 216/216 equivalence cells at max abs error 4.8e-7; isolated operator 18.63× and integrated module 1.089× at `B=1, T=1024`, CPU fp32, 30 paired blocks per cell. The 1.25× integration threshold was frozen before the grid ran and was not met |

### Mechanism checks behind the retained-trace run

These were verified before the study was trusted, and are the reason its
two-stage schedule is sound rather than assumed:

- carrier-then-tail with a KV cache reproduces a one-shot forward over the
  concatenation to fp16 noise (max |Δlogit| 0.035);
- injecting during stage 1 with the hook removed equals a single pass with the
  edit position-masked to carrier tokens (max |Δlogit| 0.031);
- the injection actually moves the answer position (max |Δlogit| 8.35);
- no forward hook is registered when stage 2 runs, asserted in code on every
  trial, not just in the smoke test.

Those rows now live in `adaptive-monitor-sandbox/results/` rather than `/tmp`.
They support a descriptive position-bias statement and nothing more: the
specific-minus-silent shifts are ≤0.009 for one supplied menu and must not be
interpreted as adaptation or a null, because menu orders are nuisance conditions
rather than independent samples.

## Compute stopped here, deliberately

The per-model strength calibration ran to completion on 2026-08-05 and answered
the question it was for. The remaining step, rerunning the 1.5B and 3B held-out
splits under the repaired control, was **not** run, and that is a decision
rather than an omission.

Two reasons. First, it would only repair the `random` arm: the 0.5B rerun
demonstrated that the target arm is bit-identical under repaired control seeding,
and `shuffled` already carries those runs' contrasts. No reported number depends
on it. Second, cost: the 3B sweep's final layer took roughly eight hours against
eight minutes for the previous one, because 3B pushes this machine into swap
(9.9 GB of 11 GB used at the end of the run). `models.memory_warning` exists to
predict exactly this. A 3B held-out run across strengths would take days here, not
the ~80 minutes a non-swapping estimate suggests.

If those reruns are wanted, do them on a machine with enough memory to hold 3B
without paging, and use per-layer strength selection for 3B rather than one value.

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
  `bd1d9239…8cbe4`, **identical to the temporary run recorded in the table above**.
  The deterministic log-prob scoring reproduces bit-for-bit across processes and
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
- **A's causal-codebook V2 is a frozen repair-confirmation after an inspected V1
  precursor.** V2 retained the model/layer/strength/labels/gates, fixed query-scale
  matching, DEV-only centering, transitive generation-source locks, validation,
  and exact intervals, and used fresh concept directions. A two-cell post-freeze
  smoke scored target 2/2 but caused no retuning or stopping decision before the
  complete run. Raw JSONL, manifest, summary, and figure are retained together.
  The analyzer verifies raw/config hashes,
  prompts, tokens, episodes, scores, and exact balance before aggregation. The
  analysis rule was protocol-frozen and the executable analyzer is checksummed in
  the summary, but its hash was not in the protocol's source lock and the
  fail-closed implementation was hardened while raw generation was running.
  Independent reconstruction matched all 40 saved values/intervals within
  `1e-12`; rerunning the analyzer reproduces the summary byte-for-byte. The model
  ran locally on CPU in float32; no different-hardware reproduction or independent
  human review exists yet.
- A green test suite must never be translated into a behavioral result, and a
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

## Layout decision, resolved

Option 1 of the three previously listed was taken: **monorepo**. All three
directories sit under one versioned root and every relative link is retained.
112 relative markdown links were checked programmatically and all resolve.

Independent repositories would have meant rewriting every cross-document link to
a commit-pinned URL and re-pinning on each push, with a reader following the
audit trail across three places. Submodules would have preserved separate
histories but shown as bare pointers in GitHub's web view, putting the audit
trail one click further away. `git subtree` gives the same history preservation
with none of that.

The originals were archived rather than deleted, at
`../_pre-monorepo-backup/`. Delete that once you are satisfied the merged tree
is complete; it is redundant with the subtree history.

## Release checklist

- [x] Review every diff and preserve the append-only correction history.
- [x] Decide whether the two LICENSE files and all new scripts/tests are intended.
- [x] Rerun the verification contract from clean committed trees.
- [x] Give legacy and replacement artifacts unmistakably different paths.
      (`retained_test_qwen05b_*` is the original run; `*_v2_*` is the rerun under
      the repaired control. `llm_agent_adaptation.json` is legacy;
      `llm_agent_exact_*` replaces it.)
- [x] Commit raw evidence for every retained aggregate, or publish it in a
      checksummed release with a stable download link. (The `.acts.pt` tensors
      remain ignored and must be attached to a release if off-GPU verification of
      the storage figures is wanted.)
- [x] Replace local sibling links with the selected repository layout.
- [x] Pin A/B commits here and in the central claim ledger.
- [ ] Add CI for tests, Ruff, formatting, and mypy.
- [ ] Push to a remote, then render every README on GitHub and check links,
      figures, commands, and license.
- [ ] Attach only claims marked application-safe in [CLAIMS.md](CLAIMS.md).
