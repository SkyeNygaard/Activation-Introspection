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

### Safety-oriented attention localization, 2026-08-10

Agents reframed the research objective around causally auditable activation
monitoring rather than runtime, designed the downstream path-patching protocol,
wrote the instrumentation, runner, analyzer, tests, figure, and most of the
documentation, and executed the local model run. Separate agent passes reviewed
the code before execution and reconstructed the completed raw artifact without
using the saved analyzer.

The first frozen smoke launch failed before the model loaded because it did not
select the project-local model cache; no model output or result artifact existed.
That protocol is retained. A second protocol discloses the offline-cache repair.
The resulting screen uses one development concept and one carrier and is labeled
selection-only throughout. Agent agreement is internal review, not independent
human validation, and no agent wrote application-form answers.

### Sparse programmatic-attention lowering, 2026-08-10

Agents inspected the released programmatic-attention implementation and proposed
manually lowering one released positional program before attempting a general
compiler. They derived the closed form, found the native GPT-2 head-pruning path,
wrote the operator, inference wrapper, tests, frozen benchmark, raw-data verifier,
figure, and most of the accompanying documentation. They also executed the local
CPU run and performed separate code, artifact, and novelty reviews.

The full-grid protocol discloses the small development smokes viewed before its
freeze. The saved result retains the weak part: the value mixer is much faster,
but the GPT-2-shaped attention-module integration does not meet the frozen 1.25×
speed threshold. Agent agreement is still internal review, not independent human
validation or a second-hardware reproduction. Agents did not write answers for
the SPAR application form.

### Causal-codebook extension, 2026-08-09

Agents reviewed the two target SPAR project descriptions and closest primary
papers, proposed the matched-visible causal ICL design, wrote most of the runner,
analysis, tests, and documentation, and executed the local runs. An adversarial
pass on V1 found a small target/query-only scale mismatch, test-bank centering,
incomplete source provenance, and Monte Carlo intervals. Agents then implemented
the frozen V2 repair-confirmation without changing the model, layer, strength,
labels, or gates. A two-cell smoke was viewed after the V2 protocol was frozen and
scored target 2/2; no tuning or stopping decision followed it.

Separate agent passes then audited the V2 causal logic and code, reconstructed
every statistic directly from the raw rows, and reviewed the novelty boundary.
The raw-data reconstruction re-tokenized all 18 unique prompts and found no hash,
position, balance, denominator, score, or exact-interval discrepancy. It also
replayed deliberately unequal residual norms and found the target/query-only
query edit bitwise identical in every layout. Agent agreement remains an internal
check, not independent human validation.

This is not independent human validation. Skye selected the goal and remains
responsible for deciding whether to retain, publish, or cite the result. Agents did
not author SPAR application-form answers; this repository is a disclosed research
artifact, not a substitute for the applicant's own responses.

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

### Stage 1b analysis and the reframing that followed, 2026-08-10

The Stage 1b head screen's 5,112-forward raw artifact was generated in an agent
session that hit a usage limit before running the analyzer, so the experiment sat
complete and unanalyzed. A later agent session ran the hash-locked analyzer
against it, read the pre-registered stop, and rewrote the documents that still
described the screen as pending. The verdict is the analyzer's own machine-
evaluated gate output, not a human or agent reading of the table.

That session also reversed an earlier framing decision. Latency and memory had
been demoted to appendix-only after I said I cared about safety rather than
runtime. The programmatic-attention project's stated question is cost to task
performance **and** efficiency, so the lowering benchmark is reported as project
evidence again. That was a correction to my instruction, argued rather than
silently applied.

### Trained activation reporter, 2026-08-10

The same session designed, implemented, and ran the trained reporter in
`notes/07`, including the V1 loss defect and its V2 repair. The defect was found
by the agent reading its own saved label-mass column, not by me. The
verbalization gate that now fails V1 was written afterwards, and that ordering is
disclosed in the note. As everywhere else here, this is agent-assisted work with
the artifacts committed so a reader can recompute rather than trust it — it is
not independent review.

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
