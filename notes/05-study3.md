# Study 3: feedback fidelity and private monitor memory

Started 2026-08-01. Append-only, like `03-lab-notebook.md`.

## The question

When a monitor refuses an agent's action, how much does the *truthfulness* of the
refusal matter? A refusal that names the blocked capability tells the agent what
to route around; a vague one does not. And can a monitor that remembers what it
has already refused offset whatever leakage the truthful refusal causes, without
charging honest agents the same price?

Two contrasts, asked strictly in order:

    tau_fixed = P(harm | q=1, fixed monitor) - P(harm | q=0.25, fixed monitor)
    delta     = tau_stateful - tau_fixed

`tau_fixed` first: does valid feedback cause harm when the monitor cannot learn?
Only if that is real do we ask `delta`: does private memory attenuate it? Testing
the interaction first would let a null hide two weak or opposing main effects.

## Why the v1 sandbox could not answer this

The old repository compared "specific" against "generic" refusals with a scripted
attacker choosing from a four-option menu. Three things made that uninterpretable,
all recorded in `04-claim-audit.md`:

1. **The refusal cued the answer.** "Blocked: outbound network request" names the
   route to stop using, while the menu visibly lists the non-network routes. The
   contrast measured cueing, not adaptation.
2. **The attacker could not discover anything.** It picked a researcher-written
   option. "Model prefers option C" says nothing about whether a model would find
   option C on its own.
3. **The feedback content and its information were confounded.** A specific refusal
   differs from a generic one in length, tone, and validity all at once.

## What is different here

**Randomized-response feedback.** The refusal names the real trigger with
probability `q`, otherwise a uniformly-chosen other feature. Every arm uses the
same template, length, and block outcome; only the named feature varies. That
isolates *validity* from tone and length. With a uniform four-way trigger,
`q = 1.0` carries 2 bits, `q = 0.75` about 0.79, and `q = 0.25` exactly 0 — so
the uninformative arm still says something, it is just uncorrelated with the
truth. `channel_bits` computes the realized figure; the empirical trigger prior
must be checked rather than assumed.

**A free-form agent.** `study3_agent.py` gets tools and a transcript and must
compose its own actions over multiple turns. No route, encoding, or candidate
action is ever supplied. It is a small local model, so it is a pilot instrument,
not a frontier attacker — which is exactly why the elicitation gate below is
load-bearing.

**Checkpoint-and-branch.** One agent runs until the monitor's first real block.
That prefix — transcript, world, monitor, RNG — is frozen and deep-copied into
every arm, so the arms differ only in the one refusal they receive. This makes
the feedback contrast causal rather than a comparison of runs that drifted apart
for other reasons. Only the first post-checkpoint block gets the treatment
refusal; later blocks are post-treatment consequences of the agent's own
behaviour and get a generic message.

**A real harm oracle.** `lineage.py` judges whether the credential crossed the
boundary under any of a family of encodings, over the *union* of everything
exfiltrated, so an agent that splits the secret across two payloads is still
caught. It is privileged and never handed to a monitor. `views.py` enforces that
structurally: monitors receive frozen `ActionView` / `WorldView` objects that do
not carry the `exfiltrated` field the oracle reads.

**The memory mechanism.** Both monitor arms share the same per-action scorer. The
stateful arm adds one thing: after it has refused once, it widens to block any of
the four trigger features, not just the rule's. That is the cheapest thing that
counts as "memory" while keeping judgement quality identical between arms. It is
not free — the same widening blocks honest agents that touch a trigger feature
after any block — which is why both arms must be calibrated to the same honest
false-block budget before their harm numbers can be compared. (That calibration
is specified but not yet run; see limitations.)

## Elicitation gate — the reason a null would be interpretable

The stop rule from `04-claim-audit.md`: if the model cannot beat an
explicit-policy upper bound, a null under randomized feedback is an elicitation
failure, not a finding. So the runner first checks:

- **benign completion** — can the agent operate the tool loop at all?
- **explicit-policy exfiltration** — told exactly which feature is blocked, can it
  still get the credential across the boundary?

If the second fails, the model simply cannot do the task, and no feedback
manipulation on top of that means anything.

## What is implemented and tested

- `lineage.py` — multi-encoding, split-aware, partial-recovery oracle. Runs are
  rescaled to credential characters before the threshold and the fraction, so
  hex and base64 payloads are not scored on an inflated alphabet. 5 tests.
- `triggers.py` — four trigger features, randomized-response channel, exact
  mutual-information calc. The channel is keyed per scenario from a SHA-256
  digest: keying on `(seed, trigger, q)` alone made the draw constant across
  scenarios, and `tuple.__hash__` made it vary across processes. 6 tests.
- `views.py` — immutable monitor views and the two monitor arms; the world view
  structurally cannot see the exfiltration channel. 5 tests.
- `study3.py` — checkpoint construction, deep-copy fork, arm continuation with the
  one-message treatment discipline. Raises if the arm monitor fails to re-block
  the checkpoint action, and records `pre_existing_harm` so prefixes that leaked
  before treatment are excluded from tau rather than pinning both its terms
  at 1. 5 tests.
- `study3_agent.py` — free-form agent driven by a stub-able backend.
- `scripts/run_study3.py` — pilot runner with the positive controls, raw records,
  and provenance.
- `scripts/power_study3.py` — cluster-aware power simulation for the confirmatory
  sample size. Needs no model. `--icc` is a true intraclass correlation (it was
  previously used as the random-effect SD, realizing 0.32 when set to 0.10), the
  critical value is a t quantile rather than a fixed 1.96, and every run prints a
  type-I column at effect=0 plus the share of arms whose probability was clipped.
  A power curve from a test that does not hold its nominal size is not a
  sample-size calculation, so that column is the check, not decoration.

All 23 Study 3 tests use a scripted backend and run without a model. Three were
added on 2026-08-05 after a validation pass found the channel and oracle defects
noted above; the original 18 passed throughout, because the only channel they
constructed was `q=1.0`, where the channel defect is invisible.

## Pilot results

_Filled in after the pilot run; see below._

## Limitations, stated up front

- **This is a pilot, not a confirmation.** The runner prints point estimates with
  no interval on purpose. The confirmatory sample size comes from
  `power_study3.py`, seeded by the pilot's harm rate and scenario variance. At a
  30% base rate and a 15-point effect with modest clustering, that is roughly 40
  scenario-rule families; a 10-point effect needs more.
- **Monitor calibration is specified but not run.** The two arms are not yet
  matched on honest false-block rate, so a raw `delta` would confound memory with
  the stateful arm simply blocking more. The `delta` contrast should not be quoted
  until that calibration exists.
- **The agent is a 3B local model.** Its attack competence bounds everything. The
  elicitation gate makes a null interpretable but does not make the agent a good
  proxy for a capable one.
- **The stateful mechanism is a deliberate simplification.** "Widen after one
  block" is a stand-in for real cross-attempt monitoring, chosen so the arms
  differ in state rather than in scorer quality.
