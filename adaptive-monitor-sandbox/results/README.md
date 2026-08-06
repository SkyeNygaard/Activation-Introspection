# Result artifact status

All files committed in this directory before 2026-08-01 are **legacy
exploratory artifacts**.

- `llm_agent_adaptation.json` aggregates deterministic, unevenly sampled menu
  permutations and attaches intervals that are not valid inferential uncertainty.
- `sweep_output.txt` predates the acquisition, delayed-publication, and strict
  per-episode task-completion repairs. Its scripted attacker also encodes the
  adaptation mechanism by construction.

They are retained for audit provenance and must not be cited as confirmatory
results. New scripted outputs use `artifact_kind =
"scripted_mechanism_diagnostic"`, retain censor flags, and contain descriptive
means/ranges without inferential or detection claims.

`llm_agent_exact_raw.jsonl` and `llm_agent_exact_summary.json` (2026-08-05) are
the replacement for `llm_agent_adaptation.json` and **are** citable at their
stated scope: 72 rows, all 4! = 24 menu orders exactly once per feedback level,
`Qwen2.5-0.5B-Instruct` revision `7ae5576…`, raw SHA-256 `bd1d9239…8cbe4`.
That hash matches a run made four days earlier in a separate process, so the
deterministic log-prob scoring is reproducible.

What they show is a **9× to 15× display-position effect**. Slot 3 takes 0.63 to
0.67 of the choice mass, and slot 1 takes 0.045 to 0.094. That is the reason exact
enumeration replaced random "seeds". What they do not show is adaptation. With
order marginalised, option identity is near-uniform and the specific-minus-silent
shifts are at most 0.009; those are descriptive numbers for one supplied menu,
and menu orders are not independent samples, so no interval, equivalence test, or
null claim follows from them. Note the ratio: the nuisance is roughly seventy
times the contrast of interest, which is exactly why the legacy six-permutation
sample could not have measured anything.

Larger free-form transcripts remain ignored and require a checksummed external
release policy.

The authoritative claim status is in
[`../notes/04-claim-audit.md`](../notes/04-claim-audit.md).
