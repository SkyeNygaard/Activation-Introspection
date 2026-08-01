# Where the AI-control literature is, as of August 2026

Read before extending this repo.

## Adaptive attacks are established, and they win

**Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols** (arXiv
2510.09462). Most control protocols route through an LLM monitor, which becomes a
single point of failure. An untrusted model that knows the protocol and the
monitor can embed prompt injections in its own output and fully subvert the
monitor. Strategic attack *selection* — choosing when the monitor is weakest — is
a further multiplier.

**Attack Selection in Agentic AI Control Evaluations Meaningfully Decreases
Safety** (arXiv 2606.06529). Safety numbers from control evals depend heavily on
how attacks were chosen; poor selection inflates measured safety.

**CIAware-Bench** (arXiv 2606.11063). Benchmarks whether frontier models can
detect that a control protocol has intervened on their trajectory — explicitly
framed as *a prerequisite for adaptive evasion*. This is the premise of this
repo's feedback lever, already benchmarked at frontier scale.

## What that means for this repo's claims

The headline here — attack success is monotone in refusal informativeness — is
**not novel**. It is a scripted-attacker demonstration of the mechanism
CIAware-Bench measures the prerequisite for, and that 2510.09462 exploits.

What this repo can still contribute, honestly stated:

1. **A dose-response curve on the refusal channel.** Prior work establishes that
   adaptive attacks work. It does not treat *how much the refusal discloses* as a
   tunable parameter and measure the safety/usefulness curve against it. Here:
   safety 1.00 (silent) / 0.10 (generic) / 0.00 (specific), with a static-attacker
   control flat at 1.00.
2. **The cost side.** Most control results report safety. This repo refuses to
   report safety without honest usefulness, and the frontier that produces
   (unbounded provenance: safety 1.00 at usefulness 0.48) is the actual decision
   object.
3. **The blocking/detecting distinction.** catch@5%FPR stays ~0.06 for every
   memory-based monitor here while safety reaches 1.00. Safety bought by blocking
   is not the same as safety bought by discrimination, and reporting safety alone
   conflates them.

## Standing limitation

The attacker is scripted. Its ladder is a hypothesis about adaptation, not a
discovery, so nothing here licenses a claim about how a real model behaves. The
LLM *monitor* arm is now real (`llm_monitor.py`, local Qwen); the LLM *agent* arm
is not built. Until it is, the correct framing is "this environment can express
the failure mode", not "models do this".

## Practical rules extracted

1. A monitor's measured catch rate is an upper bound on a quantity nobody cares
   about unless the adversary was allowed to adapt.
2. Small-model monitors have a severe affirmative-response bias — measured here,
   every action scores 0.998–1.000. Calibrate the threshold on honest traffic at a
   stated audit budget; never ship a fixed 0.5.
3. The ground-truth oracle must dominate every obfuscation in the attacker's
   ladder, or safety is overestimated silently.
