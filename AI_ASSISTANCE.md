# AI assistance disclosure

Large-language-model agents were used heavily across this repository. They
proposed and wrote code, drafted prose, ran adversarial review passes over the
apparatus, and executed experiments.

**Agent agreement is not independent verification.** Several of the errors
recorded in [`spar-application/CLAIMS.md`](spar-application/CLAIMS.md) were found
by an agent reviewing code that an agent had helped write. That is a useful check.
It is not an independent one.

Skye Nygaard is the human author, and remains responsible for every public claim,
result, artifact selection, and correction here.

This mirrors the policy in the companion [ARC White-Box Estimation Challenge
repository](https://github.com/SkyeNygaard/AI-Safety-Roadmap), which carries the
same disclosure.

## What was agent-assisted, specifically

The validation pass on 2026-08-05 was largely agent-driven. It did the following.

It re-derived every headline number in the retained-trace study from the raw
per-trial rows, without using the saved aggregates.

It found four problems in that study. Three of the four control arms were
arithmetic identities. The `random` arm gave every concept a byte-identical edit.
The storage result needed a propagation control that it did not have.

It found three more in Study 3. The feedback channel never randomized. The harm
oracle was roughly twice as sensitive to hex-encoded payloads as to plaintext. The
power simulation's `--icc` argument was not an intraclass correlation.

It then wrote the source repairs, the tests that pin them, and
`analyze_retained.propagation_control` with its figure. It ran the repaired-control
rerun, the exact-order menu diagnostic, and the per-model strength calibration
sweeps. It drafted or rewrote much of the prose in `CLAIMS.md`,
`AUDIT-MANIFEST.md`, both repository READMEs, `notes/05-retained-trace.md`, and
this file. It performed the monorepo restructure and wrote the commit messages.

Earlier work in both code repositories was agent-assisted throughout as well.

## What that does and does not license

The numbers can be reproduced from the committed raw rows and the committed code.
That is a property of the artifacts, not of who typed them. It is also why the raw
JSONL and the checksums are committed rather than summarised. A reader who
distrusts the process can rerun the verification contract in
[`spar-application/AUDIT-MANIFEST.md`](spar-application/AUDIT-MANIFEST.md) and
recompute every table from `results/`.

What agent assistance does not supply is an independent referee. Two things are
still outstanding, and they are marked as such rather than quietly assumed:

- No named human has reviewed the statistical design who did not also write it.
- No independent reproduction has been run on different hardware.

Agent reports are kept as provenance where they exist. Read them as research notes
or audit artifacts. Do not read them as independent verification.
