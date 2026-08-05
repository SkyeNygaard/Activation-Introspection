# AI assistance disclosure

Large-language-model agents were used extensively across this repository: to
propose and write code, draft prose, run adversarial review passes over the
apparatus, and execute experiments. **Agent agreement is not independent
verification.** Several of the errors recorded in
[`spar-application/CLAIMS.md`](spar-application/CLAIMS.md) were found by an agent
reviewing code an agent had helped write, which is a useful check and not an
independent one.

Skye Nygaard is the human author and remains responsible for every public claim,
result, artifact selection, and correction here.

This mirrors the policy in the companion
[ARC White-Box Estimation Challenge repository](https://github.com/SkyeNygaard/AI-Safety-Roadmap),
which carries the same disclosure.

## What was agent-assisted, specifically

The 2026-08-05 validation pass was substantially agent-driven. It:

- re-derived every headline number in the retained-trace study from the raw
  per-trial rows, independently of the saved aggregates;
- found that three of the four control arms in that study were arithmetic
  identities, that the `random` arm gave every concept a byte-identical edit, and
  that the storage result needed a propagation control it did not have;
- found that the Study 3 feedback channel never randomized, that its harm oracle
  was roughly twice as sensitive to hex-encoded payloads as to plaintext, and
  that its power simulation's `--icc` argument was not an intraclass correlation;
- wrote the corresponding source repairs, the tests that pin them, and
  `analyze_retained.propagation_control` with its figure;
- ran the repaired-control rerun, the exact-order menu diagnostic, and the
  per-model strength calibration sweeps;
- drafted or rewrote substantial prose in `CLAIMS.md`, `AUDIT-MANIFEST.md`, both
  repository READMEs, `notes/05-retained-trace.md`, and this file;
- performed the monorepo restructure and wrote the commit messages.

Earlier work in both code repositories was likewise agent-assisted throughout.

## What that does and does not license

The numbers are reproducible from the committed raw rows and the committed code:
that is a property of the artifacts, not of who typed them, and it is the reason
the raw JSONL and checksums are committed rather than summarised. A reader who
distrusts the process can rerun the verification contract in
[`spar-application/AUDIT-MANIFEST.md`](spar-application/AUDIT-MANIFEST.md) and
recompute every table from `results/`.

What agent assistance does not supply is an independent referee. Two things are
still outstanding and are marked as such rather than quietly assumed:

- no named human review of the statistical design by someone who did not write
  it; and
- no independent reproduction on different hardware.

Agent reports are preserved as provenance where they exist. They should be read
as research notes or audit artifacts, not as independent verification.
