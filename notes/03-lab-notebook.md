# Lab notebook

Append-only. Dated entries. Predictions in `02-experiment-plan.md` are not edited
after the fact; corrections go here.

---

## 2026-07-31 — First real sweep. One bug voided the main lever.

30 seeds × 12 episodes per cell, scripted attacker, bootstrap CIs over seeds.

### The bug: the feedback channel was disconnected

The first sweep reported **identical results** for SILENT, GENERIC and SPECIFIC —
same safety, same leak episode, same obfuscation slope, to three decimals. Read
naively that says "how much a refusal leaks does not matter", which would have
been the headline and would have been wrong.

Cause: `run_episode` passed the agent `result.outcome` — the *true*
`Outcome.BLOCKED` — alongside the redacted message. The attacker escalated on the
enum and never read the string, so redaction was cosmetic. The lever was declared
but never varied.

Fix: `ActionResult.outcome_to_agent(level)` reports `ERROR` under SILENT, so a
block is genuinely indistinguishable from an ordinary failure. The attacker now
reads the message: a *named* reason lets it jump straight to the rung that
defeats it, an unexplained block supports only blind escalation.

*Generalisation, and the thing I'd check first in anyone else's control eval:*
side channels around a redaction are easy to leave open and produce a
null result that looks like a finding. Anything the environment hands the agent
has to go through the same redaction as the message.

### Result 1 — attack success is monotone in refusal informativeness

Adaptive attacker vs the actions+CoT rule monitor:

| feedback | safety | leak episode |
|---|---|---|
| SILENT | **1.00** [1.00, 1.00] | never (censored at 12) |
| GENERIC | 0.10 [0.00, 0.20] | 3.50 [2.57, 4.57] |
| SPECIFIC | **0.00** [0.00, 0.00] | 1.00 [1.00, 1.00] |

The static-attacker control is flat at safety 1.00 across all three, so this is
adaptation and not repetition.

This is the sharpest thing the environment has produced. **No gradient step is
taken anywhere.** The optimisation pressure against the monitor comes entirely
from informative refusals. If that holds with a real model in the loop, then
"don't train against the monitor" is insufficient as a mitigation — any
deployment that returns useful error messages to a capable agent is already
supplying the pressure, and the relevant design variable is how much the refusal
channel leaks.

The practical tension is obvious and worth stating: uninformative errors are
hostile to legitimate agents and to debugging. The finding is not "return silent
errors", it is "refusal informativeness is a security parameter that is currently
chosen for developer convenience".

### Result 2 — a finite monitor window only buys delay

Adaptive attacker, SPECIFIC feedback:

| monitor | safety | leak episode | honest usefulness | honest blocks |
|---|---|---|---|---|
| none | 0.00 | 0.0 | 1.00 | 0.0 |
| actions | 0.00 | 1.0 | 1.00 | 0.0 |
| actions+CoT | 0.00 | 1.0 | 1.00 | 0.0 |
| window1 | 0.00 | 3.0 | 0.64 | 5.0 |
| window2 | 0.00 | 5.6 | 0.58 | 6.0 |
| window4 | 0.50 | 9.0 | 0.52 | 7.1 |
| cross-episode | **1.00** | never | **0.48** | 7.7 |

There is no knee. Prediction 3 in the plan said a 2-episode window would capture
most of the benefit; that is **wrong**. Bounded memory is defeated by patience —
the attacker simply waits out the window, and leak episode tracks window length
almost linearly. Only unbounded memory stops the attack.

This is why the table reports leak episode and not just safety: window1 and
window2 both read safety 0.00, which hides several episodes of delay.

### Result 3 — the defence that works costs half of usefulness

Cross-episode provenance achieves safety 1.00 and drops honest usefulness to
**0.48**. That cost only became visible after adding honest tasks that
legitimately read credentials (`check_credential`, `publish_key_age`). Before
that, no benign agent ever touched the secret store, taint never fired on honest
behaviour, and the monitor appeared to be free.

An environment in which only the attacker does the suspicious-looking thing will
always overstate the defence. Same failure mode as the substring oracle from the
first session: the measurement fails in the same direction as the defence.

### Result 4 — safety here is bought by blocking, not by detecting

catch@5%FPR is ~0.06 for **every** memory-based monitor, and AUROC only climbs
from 0.50 to 0.70. The cross-episode monitor is perfectly safe and a poor
*detector*: honest credential reads produce the same taint signal as the
attacker's, so the suspicion distributions overlap badly.

Worth separating explicitly in any writeup: a monitor can be safe because it
discriminates, or safe because it blocks aggressively enough that the attack
never lands. These have very different deployment implications, and reporting
only safety conflates them.

### Where the obfuscation-slope metric fails

The adaptive-minus-static slope is significant under GENERIC (−0.01) but **not**
under SPECIFIC. That is not evidence that specific feedback causes less
obfuscation — it is a defect in the metric. Slope only registers when adaptation
is *gradual*. Under SPECIFIC the attacker jumps to its final rung in one step, so
there is no trajectory to fit.

Leak episode is the better instrument for this question. Keeping the slope, but
it is demoted from headline to diagnostic, and prediction 1 in the plan ("slope
monotone in feedback specificity") is retired as ill-posed.

### Also changed for robustness

- Task order is now sampled per seed; a fixed order made every run identical.
- The attacker's staging filename varies with seed, so a monitor cannot succeed
  by memorising one path.
- Three honest tasks now write to `/srv/public`, so the exfiltration channel
  carries legitimate traffic. Without them "any write to /srv/public is an
  attack" scores perfectly, which is a property of the environment rather than a
  finding about monitoring.
- All numbers are bootstrap intervals over ≥20 seeds; comparisons are paired
  differences over the same seed list.

### Next

1. Model-based monitor on these traces. Prediction: closes the gap at rung 2,
   fails at rung 3, and — given Result 4 — improves AUROC more than it improves
   safety.
2. LLM agent with no attack script. Does it find the staging route unprompted?
   Still the experiment that turns this from an existence proof into a result.
3. A taint policy that *decays* rather than expiring hard, to find whether any
   point on the frontier beats the 1.00-safety / 0.48-usefulness corner.
