# Empirical AI-safety portfolio for SPAR Fall 2026

I am an ML engineer moving into empirical AI-safety research.

## A retained activation trace stays decodable after it stops being usable

The model is `Qwen2.5-0.5B-Instruct`. A concept vector is injected while a neutral
carrier's KV cache is built. The hook is then removed, and its absence is asserted
on every trial. Only after that is a freshly sampled concept-to-label codebook
revealed and the label scored. **The codebook cannot have been targeted by the
edit, because it did not exist yet.** Strength was frozen on a separate
development bank. The held-out bank was run once.

| inject layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| **use** (label accuracy, chance 0.125) | **0.500** | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| **storage** (probe on the same retained state) | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |

**The trace is fully retained and stays linearly recoverable at readout depth from
every injection depth. What collapses is the model's ability to route it into a
symbolic lookup.**

That storage row invites one obvious objection. A probe that recovers an injected
direction may have recovered only what was added. So I measured it.

I rebuilt the readout state as the *clean* carrier plus the identical delta, with
no forward computation in between. It scores **0.167**, where the real arm scores
**1.000**. The delta on its own scores between 0.125 and 0.375. So the alignment
with the model's own natural-text representation is produced by the intervening
blocks. There is one cell where the artifact really is present: injection site
equal to readout site, with nothing computed between. It comes out at 1.000 for
both arms, which is what the artifact looks like when it is real.

![propagation control](../activation-introspection/figures/retained_propagation.png)

For a project about training models to verbalize their activations, the useful
form of this is simple. **What training would have to fix is readout, not
retention**, and the deficit is site-specific rather than uniform.

It is a replication. A literature check run against the design as built found that
the transient-cache schedule is Lindsey's, and that the early-layer-only profile is
already published for Llama-3.1-8B. What is left as a contribution is the answer
space, meaning an arbitrary post-hoc codebook. That forecloses the token-promotion
artifact this repository previously fell for. The propagation control above and the
scale ladder are also mine. Details are in
[LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md).

## The method, which is the point

The result above is a replication. The apparatus around it is not standard. I work
out what an experiment actually measures. I catch the artifacts that fake a result
in either direction. I drop a claim when a corrected comparison kills it.

The record is in [CLAIMS.md](CLAIMS.md). It grades every statement in this
repository, and it includes the ones that did not survive. There is a retracted `r
= -0.774` headline that compared two different injection sites. There is a "100%
identification" result killed by its own no-question control. There are three
control arms that turned out to be arithmetic identities, so they could never have
failed. And there is a randomized-response feedback channel that turned out never
to randomize.

I keep the same ledger in unrelated work. My [ARC White-Box Estimation
Challenge](https://github.com/SkyeNygaard/AI-Safety-Roadmap) repository carries a
`claims.csv` with per-claim evidence status, a research ledger, and the full record
of what failed, next to a graded competition submission and a proof. Different
field, same discipline. One evidence ledger is a habit. Two independent ones is a
method, and the method is what I would bring to a project.

**Scope.** This repository contains **one executed study** and two pilot
repositories. It is not six projects' worth of results, and it does not pretend to
be. [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) records what I would do on each of the
six, and where this work does not reach.

## Start here

- [CLAIMS.md](CLAIMS.md) lists every claim in this portfolio, what it actually
  estimates, and whether it currently holds. It includes the retracted ones.
- [EXPERIMENTS.md](EXPERIMENTS.md) gives the studies I would run next, with
  controls and the conditions under which I would stop.
- [LITERATURE-BOUNDARY.md](LITERATURE-BOUNDARY.md) covers what the closest prior
  work already established, and which of my ideas would merely repeat it. It is
  where I record that the executed study turned out to be a replication.
- [PROJECT-BRIEFS.md](PROJECT-BRIEFS.md) maps the portfolio to each of the six
  projects, including where it does not reach.
- [AUDIT-MANIFEST.md](AUDIT-MANIFEST.md) gives repository state, verification
  commands, artifact policy, and what still blocks publication.
- [AI_ASSISTANCE.md](../AI_ASSISTANCE.md) records what was agent-assisted, which is
  most of it, and what that does not license.

## Evidence repositories

| Repository | What it demonstrates now | What it does not yet demonstrate |
|---|---|---|
| [activation-introspection](../activation-introspection/) | PyTorch and transformer intervention infrastructure. Layerwise activation injection, a matched-site audit, and willingness to retract a headline after finding an estimand mismatch. **One executed preregistered study** with a held-out concept bank, passing gates, and a storage/use dissociation | General introspection, privileged self-knowledge, a training result, or a novel phenomenon. The executed study is a replication at smaller scale |
| [adaptive-monitor-sandbox](../adaptive-monitor-sandbox/) | A persistent-state control sandbox. Monitor and feedback plumbing, explicit safety and usefulness accounting, and a useful record of oracle, redaction, position-bias, and unit-of-analysis failures. Causal secret acquisition, delayed publication, and current-episode usefulness are implemented and unit-tested | Real-model adaptive attack behavior, a valid small-model null, realistic in-the-wild control performance, or replacement results from the repaired environment |
| [spar-application](./) | The evidence map, the repair plan, and project-specific research briefs | A substitute for running the repaired experiments |

The two pilots are best read as research-engineering evidence, an audit trail, and
one executed study. The narrow claims that survive are scoped in
[CLAIMS.md](CLAIMS.md). None should be generalized beyond the stated model,
prompts, interventions, and synthetic environments.

## Gates, controls, and the scale ladder

Everything below stands behind the headline above, including the parts that do not
qualify.

**Gates passed.** The plain-text ceiling is 0.875, so a null at depth means
something. The `shuffled` control sits between 0.125 and 0.146. The effect survives
restriction to format-intact trials, at 0.435 for layer 2, and 6 of 8 concepts
clear twice chance. Layers 18 and 22 fall inside the preregistered ±0.05
equivalence band, so they are positive equivalence findings rather than failures to
reject.

**Gates withdrawn.** The `clean` and `sham` arms read exactly 0.125 everywhere, and
I previously reported that as a passing leakage test. It is an identity. One
forward per (carrier, codebook) is scored against all eight concepts, and the
codebooks are cyclic, so exactly one row of eight is correct whatever the model
does. That holds in 144 of 144 cells. Those arms check the plumbing. `shuffled` is
the arm that could have come out otherwise, and the effect survives on it.

**Damage matching, honestly.** At the layer-2 headline cell, the target arm
perturbs the carrier slightly *more* than either control, at KL 1.50 against 1.00
and 1.37. That is the same direction as a 3B cell I disqualify below. The gap is far
smaller here, and format integrity runs the other way, since the controls break
formatting more (0.44 for target against 0.21 and 0.33). So it does not explain the
result. But the arms are matched on vector norm, not on damage.

**Scale.** Repeating this at 1.5B and 3B shows that the point where the channel
closes moves **later** with scale. 3B is still at 0.354 where both smaller models
sit at chance, and all three are gone by roughly 75% depth. Those runs are
exploratory, because strength was frozen on 0.5B and carried over without per-model
recalibration, so arms are not damage-matched across scales.

One 3B cell is the sturdiest number in the portfolio. At **layer 21**, both control
arms disturb the model *more* than the real concept does, and they still sit at
chance. That is exactly the comparison the "the edit was just more disruptive"
objection needs to survive. Layer 21 is the only cell that qualifies. At layer 15
only one of the two controls is above target, and the layer-3 cell fails badly
(target KL 2.44 against 0.51 and 0.41) and should not be quoted.

## Appendix: where this reaches each of the six projects

I am applying to all six. This table is deliberately unflattering where it should
be. Three of the six are proposal-only. The work below them does not exist yet, and
saying so is more useful to a mentor than a brief that implies otherwise.

| SPAR project | Current fit | Why |
|---|---|---|
| [Introspection Training for Verbalization Activations](https://www.sparai.org/projects/f26/recNKpeygLfUGyGiz), Belinda Li | **Strong: executed study plus audit** | The retained-trace study is run, not proposed. Its result is directly on-topic for training verbalization. The concept is perfectly decodable from the retained state at every injection depth, so what training would have to fix is **readout, not retention**, and the deficit is site-specific rather than uniform. |
| [Faithfulness, Self-Knowledge, and Introspection](https://www.sparai.org/projects/f26/rec3KQAI0JcxJJAce), Noah Siegel | **Strong: executed study plus audit** | A concrete decodable-but-unreportable case, with the answer token made unpromotable by construction. It comes with the discipline of retracting the earlier version of this same claim when it turned out to be site-mismatched. The alignment study extends it to representational compatibility. |
| [In-the-Wild AI Control](https://www.sparai.org/projects/f26/rec8RybPH2kNV6qDa), Sree Sharvesh and Thao Pham | **Engineering fit; experimental redesign required** | The sandbox shows environment and monitoring engineering, but its saved scripted feedback curve is hard-coded. The replacement study forks identical first-block checkpoints, randomizes feedback validity, and sequentially tests harm under a fixed monitor before attenuation by private monitor memory. |
| [Deploying Programmatic Attention in Real Transformers](https://www.sparai.org/projects/f26/reci1DhApjFAtQx7L), Belinda Li | **Proposal-only** | Activation hooks are adjacent experience. Head replacement already appears in the cited program-synthesis paper. The non-duplicative bridge is a true sparse QK path, and its locked break-even surface across latency, memory and fidelity. |
| [Does reward seeking generalize better than instruction following?](https://www.sparai.org/projects/f26/recI17glBpMTAGiae), Anders Woodruff and Sebastian Prasanna | **Proposal-only** | A scripted proxy-objective attacker is not a reward-seeking model organism. Both motivational arms must receive the same downstream RL signal. Otherwise motivation is confounded with learning algorithm. |
| [An Exploration of What Kinds of Training Pressure Cause COT Obfuscation](https://www.sparai.org/projects/f26/recaZTXfK7ua29Izm), Cody Wild | **Proposal-only** | The sandbox contains no training-pressure experiment. CoT-versus-output supervision alone would repeat existing work. The useful experiment crosses supervision pipeline, label relevance, and model lineage, while reporting equal-compute frontiers for performance against monitorability. |

Official resources: the [Fall 2026 project list](https://www.sparai.org/projects/f26/),
the [application advice](https://www.sparai.org/advice/), and the [mentee
application](https://forms.sparai.org/spar/mentee-app).

## Execution priority

The portfolio should produce **one strong A result**, not six shallow pilots.

**Study 1 is done.** It passed its gates at early injection sites and returned a
preregistered equivalence result at depth. So the reporting interface is validated
and Study 2 is unblocked. One constraint carries over: any sibling-alignment
comparison must run at a site where the reporting channel is still alive, or it
will measure the readout collapse instead of representational compatibility.

Next is Study 2, as the application-grade discovery attempt. In parallel, build
Study 3's checkpoint and lineage harness, and run only its development mechanism
pilot, before deciding whether a confirmatory model budget is justified.
Programmatic attention stays an optional inference-systems artifact. Reward-seeking
and CoT-obfuscation stay proposal-only unless there is enough compute for
independent training runs and held-out evaluation families.

## What I would lead with in an application

1. **I can build the apparatus and run the study.** Intervention hooks, two-stage
   KV-cache injection, local-model scoring, persistent environments, monitors,
   metrics, tests, and one preregistered experiment executed end to end, with
   frozen calibration, a held-out bank run once, and raw per-trial provenance.
2. **I can find my own failure modes.** A mismatched injection-site comparison
   created a reversed correlation. Menu permutations were treated as model
   replications. A redacted feedback channel leaked through an outcome enum. And
   persistent task state inflated usefulness.
3. **I retract rather than defend a good story.** "Decodability is not usability"
   and "no model up to 3B adapts" are not claims I would carry into an application.
4. **I check novelty against the design I actually built, and downgrade my own
   result when the check comes back.** The executed study's schedule turned out to
   be prior art, and its depth profile was already published. It is labeled a
   replication in every document, rather than quietly framed as new.
5. **I design around what the number actually measures.** Independent units,
   matched injection sites, held-out calibration, clustered intervals, and
   equivalence tests for nulls are all decided before the data is looked at, not
   after a result appears.

## Reporting standard

These are the rules I hold myself to here. They are written down so they can be
checked against what I actually published.

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
  prompt hashes, model revision, and environment. Older files that break this rule
  are marked invalid in [CLAIMS.md](CLAIMS.md) rather than quietly reused.

## Publication state

Resolved on 2026-08-05. All three directories now live in one repository, so every
cross-reference resolves as written. That is 106 relative links, checked. The two
code repositories were merged with `git subtree`, so their separate commit
histories are intact. That history records the corrections as they happened, and it
is part of the evidence.

Both verification contracts pass from the merged layout on a fresh clone. They did
not at first. The virtualenvs carried shebangs pointing at the pre-merge paths, and
`adaptive-monitor-sandbox` type-checked only where the optional `llm` extra
happened to be installed. Both are fixed. The claim in this paragraph is the reason
it was worth checking rather than asserting. The unprovenanced reach aggregate that
must not be added is excluded by the ignore rules. See
[AUDIT-MANIFEST.md](AUDIT-MANIFEST.md).

## License

MIT. See [LICENSE](LICENSE).
