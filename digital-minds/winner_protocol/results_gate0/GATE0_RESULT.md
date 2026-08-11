# Gate 0 — the vGOLD direction is causally active; the protocol's readout is not

Run 2026-08-10 on `Qwen/Qwen3-4B-Instruct-2507` (bf16, MPS, M4/24 GB) with the
published concept vector `davidafrica/functional-wellbeing`,
`concept_vectors/qwen3-4b_step400/goal/mean_diff.pt`, shape (1, 36, 2560).

Artifacts: `gate0_diag.json` (72 rows), `dev_L21.jsonl` (24 episodes).

## FACTS

**The single-position edit does nothing.** Editing only the assistant-marker
position, across 3 layers × 6 factors × both signs (36 cells), moved the
next-token distribution by a maximum total variation of **3.85e-05**. The rating
reads 5.0000 in all 36. At factor 32 — where the *same* vector applied to every
position drives TV to 0.99997 and wrecks the output — the one-position version
still moves nothing.

**The vector itself works.** All-position steering at L21, factor 2:
TV **0.587**, rating **5.000 → 6.142**, at a perturbation of 8.8% of the mean
residual norm. So the published direction is causally live at a usable magnitude.

**The prescribed cell is inert.** `MODEL_FRONTIER.md` says start at L29/factor 2.
There, all-position steering gives TV 0.00006 (−vGOLD) and 0.00000 (+vGOLD).
L21 is the live layer, not L29. Layer choice is partly a magnitude choice: the
vector norm grows 5.98 → 11.13 → 13.57 from L21 to L29 while the residual norm
grows 135 → 185 → 264, so a fixed factor is a different intervention at each
depth.

**The published ordering holds only under distortion.** −vGOLD < baseline <
+vGOLD requires factor 16 (L21: 2.939 / 5.000 / 8.999), where TV is 0.93–1.00 and
the output is destroyed. At factor 2, where the intervention is non-destructive,
the effect is one-sided: +vGOLD moves the rating, −vGOLD gives TV 0.00001.

**The readout has no dynamic range.** Baseline p(top digit) = 0.99997, digit mass
1.0, entropy 0.00056 bits. The digit-logit expectation is a step function; it
cannot register anything short of an argmax flip.

**The real codebook cell confirms it** (24 episodes, 1 carrier, neutral persona,
L21, factor 2 — the layer that demonstrably works):

| arm | p(correct) | accuracy | label mass | option logprobs |
|---|---|---|---|---|
| clean | 0.500000 | 0.500 | 1.0000 | [0.0, −24.25] |
| query_only | 0.500000 | 0.500 | 1.0000 | [0.0, −24.25] |
| target | 0.499972 | 0.500 | 1.0000 | [0.0, −24.25] |

`target − query_only = −0.000028`, against a preregistered gate of **≥ +0.10**.
Semantic rating 8.9999, sd 0.000124.

Note what the 0.500 is. The predicted label is `Q` in every episode of every arm,
with a 24.25-nat gap to `K`. The correct label alternates, so 50% is arithmetic
from a constant response, not an uninformative-but-varying readout. And
`|target − clean| = 0.000028` means the intervention barely perturbs the answer
at all — this is a propagation failure, not a failure to learn the mapping.

## INTERPRETATION

Narrow claim: on this model, a published causal welfare direction is active under
broad steering, and a transient single-token edit of that same direction does not
propagate to a structured downstream readout.

Strongest confound: one carrier, one persona, one layer for the codebook cell.
That is ample to rule out a +0.10 effect when the observed one is 3e-5, but it is
not a layer sweep — L21 and L25 have live all-position effects that were not
tested with the codebook readout at higher factors.

## LESSON

The parent assumption of the winner protocol — that a transient welfare state
imposed at one position can be read out downstream — is false here. This is the
protocol's own interpretation cell: *"transient propagation from the effectual
welfare layer is insufficient under this interface. This is a meaningful negative
about the measurement method; do not rescue it with a wide hyperparameter
sweep."* Its stop rule says prune.

**STATUS: PRUNE.** Five orders of magnitude is not a tuning gap.

## NEXT

Branch 19 (preference self-knowledge). Cost: about four minutes of model time to
kill the former P1.

## Two errors made getting here

The first Gate 0 smoke returned `gate0_passed: true` on a paired delta of 0.0007.
The gate tested only the *sign* of the difference with no magnitude threshold,
and pooled carriers sitting at 5.000 and 8.999 into a meaningless mean of 6.9997
— the same vacuous-gate failure this package had already criticised in Branch 04.
Fixed: `MIN_EFFECT = 0.25` on the paired within-carrier delta, plus a
majority-ordering requirement.

The first codebook DEV run printed *"Some parameters are on the meta device
because they were offloaded to the disk"*. `run_dev.py` used `device_map="auto"`,
which silently offloads when RAM is short instead of failing. That run was
discarded. `run_dev.py` now goes through the memory guard, pins `device_map=
"mps"`, and hard-errors on any meta-device parameter; the numbers above are from
the clean re-run at 15.3 GiB available.

## Sprint provenance

All of the above predates the Apart Digital Minds sprint (14–16 Aug 2026) and is
**prior work**. If any of it appears in a submission it must be labelled as such.
