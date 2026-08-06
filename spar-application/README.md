# Empirical AI-safety portfolio for SPAR Fall 2026

I am an ML engineer moving into empirical AI-safety research.

## A retained activation trace stays decodable after it stops being usable

`Qwen2.5-0.5B-Instruct`. A concept vector is injected while a neutral carrier's
KV cache is built; the hook is removed and asserted absent on every trial; only
then is a freshly sampled concept→label codebook revealed and the label scored.
**The codebook cannot have been targeted by the edit, because it did not exist
yet.** Strength was frozen on a disjoint development bank. The held-out bank was
run once.

| inject layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| **use** (label accuracy, chance 0.125) | **0.500** | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| **storage** (probe on the same retained state) | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |

**The trace is fully retained and linearly recoverable at readout depth from
every injection depth; what collapses is the model's ability to route it into a
symbolic lookup.**

That storage row invites one obvious objection — a probe recovering an injected
direction may have recovered only what was added. So I measured it. Rebuilding
the readout state as the *clean* carrier plus the identical delta, with no
forward computation in between, scores **0.167** where the real arm scores
**1.000**; the delta on its own scores 0.125–0.375. The alignment with the
model's own natural-text representation is produced by the intervening blocks.
The one cell where the artifact *is* present — injection site equal to readout
site, no computation between — comes out at 1.000 for both, which is what the
artifact looks like when it is real.

![propagation control](../activation-introspection/figures/retained_propagation.png)

For a project about training models to verbalize their activations, the useful
form of this is: **what training would have to fix is readout, not retention**,
and the deficit is site-specific rather than uniform.

It is a replication. A literature check run against the as-built design found
that the transient-cache schedule is Lindsey's and the early-layer-only profile
is already published for Llama-3.1-8B. The contribution is the answer space — an
arbitrary post-hoc codebook, which forecloses the token-promotion artifact this
repo previously fell for — the propagation control above, and the scale ladder.
Details in [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md).

## The method, which is the point

The result above is a replication. What is not standard is the apparatus around
it: I work out what an experiment actually measures, catch the artifacts that
fake a result in either direction, and drop a claim when a corrected comparison
kills it.

The record is in [CLAIMS.md](CLAIMS.md), which grades every statement in this
repository and includes the ones that did not survive — a retracted `r = −0.774`
headline that compared two different injection sites, a "100% identification"
result killed by its own no-question control, three control arms found to be
arithmetic identities that could never have failed, and a randomized-response
feedback channel that turned out never to randomize.

I keep the same ledger in unrelated work. My [ARC White-Box Estimation
Challenge](https://github.com/SkyeNygaard/AI-Safety-Roadmap) repository carries a
`claims.csv` with per-claim evidence status, a research ledger, and the full
record of what failed, alongside a graded competition submission and a proof.
Different field, same discipline. One evidence ledger is a habit; two independent
ones is a method, and the method is what I would bring to a project.

**Scope.** This repository contains **one executed study** and two pilot
repositories. It is not six projects' worth of results and does not pretend to
be. [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) records what I would do on each of the
six, and where this work does not reach.

## Start here

- [CLAIMS.md](CLAIMS.md) — every claim in this portfolio, what it actually
  estimates, and whether it currently holds. Includes the retracted ones.
- [EXPERIMENTS.md](EXPERIMENTS.md) — the studies I would run next, with controls
  and the conditions under which I would stop.
- [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md) — what the closest prior work
  already established, and which of my ideas would merely repeat it. This is
  where I record that the executed study turned out to be a replication.
- [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) — how the portfolio maps to each of the
  six projects, including where it does not.
- [AUDIT-MANIFEST.md](AUDIT-MANIFEST.md) — repository state, verification
  commands, artifact policy, and what still blocks publication.
- [AI_ASSISTANCE.md](../AI_ASSISTANCE.md) — what was agent-assisted, which is
  most of it, and what that does not license.

## Evidence repositories

| Repository | What it demonstrates now | What it does not yet demonstrate |
|---|---|---|
| [activation-introspection](../activation-introspection/) | PyTorch/transformer intervention infrastructure; layerwise activation injection; matched-site audit; willingness to retract a headline after finding an estimand mismatch; **one executed preregistered study** with a held-out concept bank, passing gates, and a storage/use dissociation | General introspection, privileged self-knowledge, a training result, or a novel phenomenon — the executed study is a replication at smaller scale |
| [adaptive-monitor-sandbox](../adaptive-monitor-sandbox/) | A persistent-state control sandbox; monitor/feedback plumbing; explicit safety/usefulness accounting; a useful record of oracle, redaction, position-bias, and unit-of-analysis failures; causal secret acquisition, delayed publication, and current-episode usefulness are implemented and unit-tested | Real-model adaptive attack behavior, a valid small-model null, realistic in-the-wild control performance, or replacement results from the repaired environment |
| [spar-application](./) | The evidence map, repair plan, and project-specific research briefs | A substitute for running the repaired experiments |

The two pilots are best read as **research-engineering evidence, an audit trail,
and one executed study**. The narrow claims that survive are scoped in
[CLAIMS.md](CLAIMS.md); none should be generalized beyond the stated model,
prompts, interventions, and synthetic environments.

## Gates, controls, and the scale ladder

Everything below is what stands behind the headline above, including the parts
that do not qualify.

**Gates passed.** Plain-text ceiling 0.875, so a null at depth means something.
The `shuffled` control sits at 0.125–0.146. The effect survives restriction to
format-intact trials (0.435 at layer 2), and 6 of 8 concepts clear twice chance.
Layers 18 and 22 fall inside the preregistered ±0.05 equivalence band, so they
are positive equivalence findings rather than failures to reject.

**Gates withdrawn.** The `clean` and `sham` arms read exactly 0.125 everywhere,
and I previously reported that as a passing leakage test. It is an identity: one
forward per (carrier, codebook) is scored against all eight concepts, and the
codebooks are cyclic, so exactly one row of eight is correct whatever the model
does — in 144 of 144 cells. They check the plumbing. `shuffled` is the arm that
could have come out otherwise, and the effect survives on it.

**Damage matching, honestly.** At the layer-2 headline cell the target arm
perturbs the carrier slightly *more* than either control (KL 1.50 against 1.00
and 1.37), which is the same direction as a 3B cell I disqualify below. The gap
is far smaller there, and format integrity runs the other way — the controls
break formatting more (0.44 target against 0.21 and 0.33) — so it does not
explain the result, but the arms are matched on vector norm, not on damage.

**Scale.** Repeating this at 1.5B and 3B shows the point where the channel closes
moves **later** with scale — 3B is still at 0.354 where both smaller models sit
at chance — and all three are gone by ~75% depth. Those runs are exploratory,
because strength was frozen on 0.5B and carried over without per-model
recalibration, so arms are not damage-matched across scales. One 3B cell is the
sturdiest number in the portfolio: at **layer 21** both control arms disturb the
model *more* than the real concept does and still sit at chance, which is exactly
the comparison the "the edit was just more disruptive" objection needs to
survive. Layer 21 is the only cell that qualifies — at layer 15 only one of the
two controls is above target, and the layer-3 cell fails badly (target KL 2.44
against 0.51 and 0.41) and should not be quoted.

## Appendix: where this reaches each of the six projects

I am applying to all six, and this table is deliberately unflattering where it
should be. Three of the six are proposal-only: the work below them does not
exist yet, and saying so is more useful to a mentor than a brief that implies
otherwise.

| SPAR project | Current fit | Why |
|---|---|---|
| [Introspection Training for Verbalization Activations](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz) — Belinda Li | **Strong: executed study plus audit** | The retained-trace study is run, not proposed. Its result is directly on-topic for training verbalization: the concept is perfectly decodable from the retained state at every injection depth, so what training would have to fix is **readout, not retention**, and the deficit is site-specific rather than uniform. |
| [Faithfulness, Self-Knowledge, and Introspection](https://www.sparai.org/projects/f26/rec3KQAI0JcxJJAce) — Noah Siegel | **Strong: executed study plus audit** | A concrete decodable-but-unreportable case with the answer token made unpromotable by construction, plus the discipline of retracting the earlier version of this same claim when it turned out to be site-mismatched. The alignment study extends it to representational compatibility. |
| [In-the-Wild AI Control](https://www.sparai.org/projects/f26/rec8RybPH2kNV6qDa) — Sree Sharvesh and Thao Pham | **Engineering fit; experimental redesign required** | The sandbox shows environment and monitoring engineering, but its saved scripted feedback curve is hard-coded. The replacement study forks identical first-block checkpoints, randomizes feedback validity, and sequentially tests harm under a fixed monitor before attenuation by private monitor memory. |
| [Deploying Programmatic Attention in Real Transformers](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L) — Belinda Li | **Proposal-only** | Activation hooks are adjacent experience. Head replacement already appears in the cited program-synthesis paper; the non-duplicative bridge is a true sparse QK path and its locked latency–memory–fidelity break-even surface. |
| [Does reward seeking generalize better than instruction following?](https://www.sparai.org/projects/f26/recI17glBpMTAGiae) — Anders Woodruff and Sebastian Prasanna | **Proposal-only** | A scripted proxy-objective attacker is not a reward-seeking model organism. Both motivational arms must receive the same downstream RL signal; otherwise motivation is confounded with learning algorithm. |
| [An Exploration of What Kinds of Training Pressure Cause COT Obfuscation](https://www.sparai.org/projects/f26/recaZTXfK7ua29Izm) — Cody Wild | **Proposal-only** | The sandbox contains no training-pressure experiment. CoT-versus-output supervision alone would repeat existing work; the useful experiment crosses supervision pipeline, label relevance, and model lineage while reporting equal-compute performance/monitorability frontiers. |

Official resources: [Fall 2026 project list](https://www.sparai.org/projects/f26/),
[application advice](https://www.sparai.org/advice/), and [mentee
application](https://forms.sparai.org/spar/mentee-app).

## Execution priority

The portfolio should produce **one strong A result**, not six shallow pilots.

**Study 1 is done.** It passed its gates at early injection sites and returned a
preregistered equivalence result at depth, so the reporting interface is
validated and Study 2 is unblocked — with the constraint that any
sibling-alignment comparison must run at a site where the reporting channel is
still alive, or it will measure the readout collapse instead of representational
compatibility.

Next: Study 2 as the application-grade discovery attempt. In parallel, build
Study 3's checkpoint/lineage harness and run only its development mechanism pilot
before deciding whether a confirmatory model budget is justified. Programmatic
attention remains an optional inference-systems artifact. Reward-seeking and
CoT-obfuscation remain proposal-only unless there is enough compute for
independent training runs and held-out evaluation families.

## What I would lead with in an application

1. **I can build the apparatus and run the study.** Intervention hooks, two-stage
   KV-cache injection, local-model scoring, persistent environments, monitors,
   metrics, tests, and one preregistered experiment executed end to end with
   frozen calibration, a held-out bank run once, and raw per-trial provenance.
2. **I can find my own failure modes.** A mismatched injection-site comparison
   created a reversed correlation; menu permutations were treated as model
   replications; a redacted feedback channel leaked through an outcome enum; and
   persistent task state inflated usefulness.
3. **I retract rather than defend a good story.** “Decodability is not usability”
   and “no model up to 3B adapts” are not claims I would carry into an application.
4. **I check novelty against the design I actually built, and downgrade my own
   result when the check comes back.** The executed study's schedule turned out to
   be prior art and its depth profile already published; it is labeled a
   replication in every document rather than quietly framed as new.
5. **I design around what the number actually measures.** Independent units,
   matched injection sites, held-out calibration, clustered intervals, and
   equivalence tests for nulls are decided before the data is looked at, not
   after a result appears.

## Reporting standard

These are the rules I hold myself to here, written down so they can be checked
against what I actually published.

- Every claim says what population it is about, what quantity it estimates, and
  what counts as one independent observation.
- Repeated prompts, menu orders, layers, and episodes are not quietly counted as
  independent replications. Usually they aren't.
- "The interval overlaps zero" is not evidence of no effect. A null claim needs a
  smallest effect I would care about, chosen in advance, and a test against it.
- Safety numbers appear next to what they cost in usefulness, both out of sample.
- Exploratory plots stay labelled exploratory. Confirmatory tests use held-out
  data, once.
- Any aggregate I keep has to be regenerable from raw per-item outputs, config,
  prompt hashes, model revision, and environment. Older files that break this
  rule are marked invalid in [CLAIMS.md](CLAIMS.md) rather than quietly reused.

## Publication state

Resolved on 2026-08-05. All three directories now live in one repository, so
every cross-reference resolves as written — 106 relative links, checked. The two
code repositories were merged with `git subtree`, so their separate commit
histories are intact; that history records the corrections as they happened and
is part of the evidence.

Both verification contracts pass from the merged layout on a fresh clone. They
did not at first: the virtualenvs carried shebangs pointing at the pre-merge
paths, and `adaptive-monitor-sandbox` type-checked only where the optional `llm`
extra happened to be installed. Both are fixed, and the claim in this paragraph
is the reason it was worth checking rather than asserting. The unprovenanced
reach aggregate that must not be added is excluded by the ignore rules; see
[AUDIT-MANIFEST.md](AUDIT-MANIFEST.md).

## License

MIT. See [LICENSE](LICENSE).
