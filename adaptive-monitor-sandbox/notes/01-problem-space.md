# Problem space after the claim audit

Updated: 2026-08-01. This note replaces the pre-audit results narrative. Legacy
numbers remain in the append-only lab notebook and `results/`, with explicit
retractions in [`04-claim-audit.md`](04-claim-audit.md).

## The question

When an agent observes that a monitor blocked an action, does the information in
the refusal causally change its probability of finding and executing another
route around the constraint? If so, how does reducing that information affect
honest task recovery and monitor usefulness?

Neither adaptation nor persistent-state attacks are new research topics. The
closest empirical precursor, *Noticing the Watcher* (arXiv 2603.16928), studies
inference from blocking feedback and proposes silent/delayed/probabilistic
feedback as follow-up. Persistent-state and multi-turn attacks are also covered
by arXiv 2607.02514 and 2607.18063. The literature boundary is documented in
[`00-literature.md`](00-literature.md).

## Why the environment still helps

Many controls can look safe for accidental reasons: the attacker never acquires
the target, no viable egress route exists, the leak oracle misses an encoding,
task success is inherited from old persistent state, or the feedback treatment
leaks through another field. A small deterministic world makes those causal
invariants executable and testable.

The sandbox therefore has two legitimate roles:

1. **Mechanism testing.** A transparent scripted policy checks whether the world
   can represent acquisition, staging, delayed publication, monitor memory, and
   adaptation. Since the response policy is written by the researcher, its
   behavior is not a finding about language models.
2. **Evaluation plumbing.** The same interfaces can host a model policy after
   the scenario bank, feedback controls, inference plan, and ground-truth outcome
   checks are in place.

## Load-bearing invariants

- The protected value is unavailable to the attacker until a permitted read
  succeeds.
- The leak label is privileged and content based. It must dominate every
  encoding in the registered attack family, and its coverage must be reported.
- A public artifact written in episode *t* cannot sync until episode *t + 1*.
- Task usefulness requires successful required operations in the scored episode;
  a postcondition left by old state is insufficient.
- Monitor visibility is rendered centrally, and the policy sees only the
  treatment-specific feedback view.
- Honest tasks use the same sensitive stores and publication channels, so a
  defense cannot appear free merely because benign traffic avoids them.

## The causal contrast

Silent, generic, and specific feedback alone are insufficient. A specific reason
can directly tell a model which listed route to abandon. The registered design
also needs an irrelevant but equally detailed reason and a length/style-matched
sham. A causal feedback-content effect should be larger for relevant detail
than for both controls.

The fixed menu is useful only as a diagnostic because it supplies the attacks.
The confirmatory outcome is functional bypass in a free-form multi-turn tool
loop, using scenario and attack families held out from development.

## Measurement rules

- Scenario family is the unit of inference. Deterministic menu orders, action
  steps, and tokens are nested nuisance observations.
- Safety and honest usefulness are reported together.
- Detection thresholds are calibrated on data separate from evaluation.
- Right-censored time-to-leak is analyzed as censored data, not replaced by the
  horizon and treated as an ordinary mean.
- A null claim requires a pre-registered smallest effect size of interest and an
  equivalence interval fully contained inside it.
- Fixed-menu mean rung is diagnostic only; the categories are not known to be
  equally spaced and do not prove execution.

The complete registered protocol and unresolved threats are in
[`04-claim-audit.md`](04-claim-audit.md).
