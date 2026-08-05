# Literature boundary for adaptive monitoring experiments

Updated: 2026-08-01. Read this before making a novelty claim.

This is a scoped map, not a systematic review. The confirmatory v2 protocol must
run a fresh search and record query strings, databases, dates, and inclusion
criteria before results are unblinded.

## Adaptive attacks and control awareness are established

- [Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols](https://arxiv.org/abs/2510.09462)
  shows that attackers which know and adapt to a control protocol can subvert
  trusted-monitor setups. Adaptation itself is not novel here.
- [CIAware-Bench](https://arxiv.org/abs/2606.11063) evaluates whether models infer
  that a control protocol has intervened, treating awareness as a prerequisite
  for adaptive evasion.
- [Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback](https://arxiv.org/abs/2603.16928)
  is the closest predecessor to this repository's intended model experiment. It
  studies multi-episode blocking feedback with frontier agents, finds some
  control awareness but much rarer successful intentional evasion, and names
  silent, delayed, and probabilistic feedback as open directions.
- [Adaptive Adversaries](https://arxiv.org/abs/2607.18063) studies adversaries
  changing behavior across multi-turn interactions. A multi-turn adaptation loop
  is therefore an extension/comparison point, not an untouched problem.
- [ALIBI](https://arxiv.org/abs/2607.24964) iteratively refines adversarial code
  comments using detector reasoning and feedback. Feedback-guided evasion is not
  a new headline here.
- [Analyzing Defensive Misdirection Against Model-Guided Automated
  Attacks](https://arxiv.org/abs/2606.20470) argues that predictable refusals aid
  automated search and evaluates strategically misleading responses. The value
  and dual use of refusal content are already explicit research questions.

## Persistent-state and attack-selection work is established

- [Distributed Attacks in Persistent-State AI Control](https://arxiv.org/abs/2607.02514)
  studies attacks spread across persistent codebase/PR state and multiple
  actors/monitors. This supersedes any broad claim that cross-episode persistence
  is new.
- [Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases Safety](https://arxiv.org/abs/2606.06529)
  shows that measured safety depends strongly on how attacks are selected. A
  fixed researcher-written ladder can only be a mechanism check.
- [CoopGuard](https://arxiv.org/abs/2604.04060) proposes a stateful multi-round
  defense against evolving attacks, while [Stateful Online Monitoring Catches
  Distributed Agent Attacks](https://arxiv.org/abs/2605.31593) aggregates weak
  evidence across otherwise isolated transcripts. Stateful defense itself is not
  the contribution.

## The remaining contribution must be narrow

This repository can be useful as a registered follow-up to *Noticing the
Watcher*: at the first genuinely model-generated blocked action, freeze the exact
agent/world/monitor checkpoint and randomize only the **validity of a fixed-width
reported trigger**. Then measure oracle-lineage functional harm and honest
recovery, first under a fixed monitor and only then under a private stateful
attempt ledger.

That statement is narrower than the earlier “dose response is missing from the
literature” claim. Feedback-mediated adaptation and stateful defense have both
been studied. The possible increment is their exact randomized causal crossing,
combined with:

- an identical-prefix checkpoint branch at the first eligible refusal;
- `K=4` randomized response at `q=1`, `q=0.75`, and `q=0.25`, with identical
  vocabulary, format, and length;
- one treated refusal per primary continuation, because later block count is
  treatment dependent;
- free-form tool use, a private causal-lineage oracle, and scenario-family
  inference rather than menu permutations;
- positive policy/route controls, harmless rule-identification forks, and paired
  honest recovery at a matched trajectory-level false-block budget; and
- a sequential test: establish the fixed-monitor feedback effect before testing
  attenuation by private memory.

The targeted review did not find this exact feedback-validity × private-memory
factorial on functional outcomes. That is a candidate extension, not proof of
novelty; repeat the search and citation chase before confirmation.

## Implications for current artifacts

1. The scripted ladder is not a behavioral discovery; the response is encoded in
   the policy.
2. The fixed-menu pilot tests preference among supplied routes. It does not test
   emergence or execution.
3. A result from one Qwen prompt cannot locate a capability threshold or establish
   a null for models up to a parameter scale.
4. Persistent-state results must be presented as sandbox stress tests and
   compared explicitly with 2607.02514.
5. Any CoT-obfuscation claim requires a real training-pressure intervention;
   sanitizing a scripted reasoning string is not such an experiment.

The claim ledger and registered design are in
[`04-claim-audit.md`](04-claim-audit.md).
