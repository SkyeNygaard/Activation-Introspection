# Retained-trace study: preregistration and results

Started 2026-08-01. This file is append-only in the same spirit as
`03-lab-notebook.md`. The preregistration section below was written **before**
the held-out concept bank was run, and has not been edited since.

## What this study fixes

Every earlier report endpoint in this repo scored an answer while the
intervention was still live. `notes/03-lab-notebook.md` records what that cost:
word-scored identification hit 1.000, and a no-question control reproduced
1.000, because a concept vector mechanically promotes its own token. The
schedule, not the metric, was the problem.

This study removes the hook before the answer space exists:

1. forward a **neutral carrier** with the concept injected at layer `L`, keeping
   the KV cache;
2. remove the hook and assert no hook is registered;
3. only then append a **freshly sampled concept -> label codebook** and the query,
   and score the label.

The codebook does not exist while the edit is live, so the edit cannot have
promoted whichever label later counts as correct. Because the assignment is
balanced by cyclic shift, chance is exactly `1/8`.

Correction (2026-08-05): this note previously called the `clean` and `sham` arms
"genuine leakage gates rather than decoration". They are not gates. Each runs one
forward per (carrier, codebook) and scores it against all eight concepts, and the
codebooks are cyclic, so exactly one of the eight rows is correct whatever the
model does — checked, and true in 144 of 144 cells. Their 0.125 confirms the
pipeline is wired as intended and nothing more. The arms that carry information
about the concept, and could therefore have failed, are `random` and `shuffled`.

## Preregistration (frozen before the held-out run)

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-0.5B-Instruct` |
| concept banks | `DEV_CONCEPTS` for calibration, `TEST_CONCEPTS` for confirmation; disjoint, pinned in `src/introspect/retained.py` |
| injection sites | layers 2, 6, 10, 14, 18, 22 of 24 |
| strength | **frozen at `alpha = 1.0`** from the dev sweep below |
| primary endpoint | post-codebook label accuracy, target arm minus the **strongest** control arm |
| controls | `clean` (no hook), `sham` (identical hook, strength 0), `random` (norm-matched), `shuffled` (coordinate-permuted) |
| ceiling | `natural` (concept stated in plain text, same carrier/codebook/query/scoring) |
| smallest effect worth caring about | 5 percentage points, chosen in advance |
| independent unit | one concept in one carrier family. Intervals are bootstrapped over those clusters. The 8 codebook permutations exist to average out label-position bias — they are not 8 extra trials, and are never resampled as if they were |
| storage measure | a multinomial probe trained **only on ordinary sentences** mentioning each concept, then applied to the retained carrier activation. It never sees an injected example |
| validity gates | concept vectors mutually distinguishable (max off-diagonal cosine < 0.5); probe's held-out-template CV above chance; probe at chance when read out below the injection site; damage reported for every arm |

"Damage" throughout means the KL divergence between the clean and edited
next-token distributions at the last carrier token — how much the edit disturbed
the model, independent of whether it carried any usable concept information.

Stop rules, unchanged from `04-claim-audit.md`: if the `natural` ceiling is not
well above chance, a null in the injected arms is **uninterpretable, not
negative**. If the storage probe is at chance, a use null localizes nothing. A
positive result licenses only the claim "causal use of a retained trace" -- not
introspection, not self-knowledge, not privileged access.

### Why alpha = 1.0 was frozen

Chosen on `DEV_CONCEPTS` only, before any held-out concept was scored. It is the
best or near-best strength at every site, sits in a moderate carrier-KL band
(~0.7-1.4), and retains the highest format integrity among the strengths that
produce any effect at all. Larger alphas buy nothing and damage the carrier:
`alpha = 8` is at or below chance everywhere.

## Dev calibration sweep

`Qwen2.5-0.5B-Instruct`, dev concepts, 8 codebooks x 3 carrier families x 8
concepts per cell. Chance = 0.125. `clean` and `sham` were exactly 0.125 in
every cell, which is an identity rather than evidence — see the correction above.
The informative floor is `shuffled`.

| layer | a=0.5 | a=1 | a=2 | a=4 | a=8 | KL at a=1 |
|---|---|---|---|---|---|---|
| 2 | 0.245 | **0.510** | 0.370 | 0.255 | 0.141 | 1.43 |
| 6 | 0.161 | **0.302** | 0.302 | 0.172 | 0.161 | 0.88 |
| 10 | 0.161 | 0.198 | **0.240** | 0.177 | 0.115 | 0.85 |
| 14 | 0.135 | 0.135 | 0.161 | 0.141 | 0.104 | 0.73 |
| 18 | 0.130 | 0.141 | 0.109 | 0.099 | 0.120 | 0.81 |
| 22 | 0.125 | 0.120 | 0.130 | 0.130 | 0.130 | 1.30 |

The pattern that motivated the confirmatory run: **usability decays
monotonically with injection depth and is at chance by roughly mid-network**,
while the same edit remains linearly decodable from the retained state.

An audit cell run earlier at L12 / `alpha = 4` makes the dissociation concrete:
the natural-text probe recovered the injected concept from the retained carrier
at **1.000** while post-codebook use sat at **0.146** against a chance of 0.125.
That cell also failed damage matching (`random` reached KL 9.17 with format
integrity 0.12 against the target arm's 5.32/0.71), which is why `alpha = 4` was
not carried forward.

## Prior art found after the run

Searching the exact design *after* executing it turned up that the schedule and
the depth profile are both established. This is recorded here rather than
quietly dropped:

- **Lindsey (2026), *Emergent Introspective Awareness in LLMs*** (arXiv
  2601.01828; transformer-circuits.pub) introduced concept injection, **and the
  removal of the steering vector before querying is his design, not mine**.
  `00-literature.md` already said so; I still wrote the schedule up as though it
  were this repo's idea. He also sweeps injection layer, with a peak around
  two-thirds depth at frontier scale.
- [Latent Introspection: Models Can Detect Prior Concept
  Injections](https://arxiv.org/abs/2602.20031) implements the identical
  transient KV-cache protocol on Qwen2.5-Coder-32B, injecting at a fixed middle
  band (layers 21–42 of 64).
- **Krasheninnikov et al., *Detecting the Disturbance*** (arXiv 2512.12411) —
  already cited in `00-literature.md` for the affirmative-shift confound — also
  reports on Llama-3.1-8B that the surviving capacities are "confined to
  early-layer injections and collapse to chance thereafter". That is the same
  profile measured here, and it is the part of that paper my own summary of it
  had missed.

So this run is a **replication in a smaller regime, plus one control the cited
work does not use**: an arbitrary concept→label codebook sampled after the edit
is removed, which makes lexical promotion of the answer token impossible. Given
that this repository previously recorded a 1.000 accuracy that was entirely
token promotion, that control is worth having even though the phenomenon is
known. The claim is a replication with a tightened answer space at 0.5B, not a
discovery.

## Held-out results

`Qwen2.5-0.5B-Instruct`, `TEST_CONCEPTS`, `alpha = 1.0`, 6912 trials, run once.
Chance 0.125. Raw rows and checksummed provenance in
`results/retained_test_qwen05b_*`.

**Gates first.** The ceiling arm (concept stated in plain text, same carrier,
codebook, query and scoring) reached **0.875**, so a null in the injected arms is
interpretable rather than a format failure. `shuffled` — the control that
carries a per-concept edit and could have come out above chance — held at
**0.125–0.146 at every site**. `clean` and `sham` also read exactly 0.125
everywhere, but by construction rather than by result; see the correction above.

### Use: post-codebook label accuracy

| inject layer | target | strongest control | effect vs strongest control, 95% CI |
|---|---|---|---|
| 2 | **0.500** | 0.125 | **+0.375 [+0.286, +0.464]** |
| 6 | 0.193 | 0.141 | +0.052 [+0.010, +0.104] |
| 10 | 0.198 | 0.130 | +0.068 [+0.010, +0.141] |
| 14 | 0.125 | 0.146 | −0.021 [−0.052, +0.005] |
| 18 | 0.130 | 0.135 | −0.005 [−0.026, +0.016] — **equivalent** |
| 22 | 0.141 | 0.130 | +0.010 [−0.016, +0.036] — **equivalent** |

Intervals are cluster bootstraps over concept x carrier family. At layers 18 and
22 the entire interval lies inside the preregistered `[-0.05, +0.05]` band, so
those are positive equivalence findings, not "failed to reject".

### Storage: natural-text probe on the same retained state

The probe is fit only on ordinary sentences mentioning each concept and never
sees an injected example. Readout at the final measured layer (22):

| inject layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| target | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |
| control | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 |

Negative control: reading out at any layer *below* the injection site gives
exactly 0.125, as it must, since the edit has not happened yet there.

The inject-22 column is excluded from the finding below. `capture` is registered
after `intervene` on the same block, so at readout 22 it reads the edited state
with no intervening computation; its 1.000 is arithmetic. Note also that storage
is not uniform across *readout* depth — the row above at readout 14 gives 0.458
for an injection at layer 2. The claim is about readout 22.

### The finding

**From every injection depth the concept is perfectly linearly decodable from the
retained carrier state at readout layer 22, while the model's ability to use that
same state collapses to chance by layer 14 of 24.** Storage is not the
bottleneck; routing the retained state into a symbolic lookup is. Both numbers
come from the same forward passes at the same sites, which is exactly the
site-matching the retracted `r = -0.774` comparison lacked.

**Propagation control (2026-08-05).** `probe.py` warns that a probe recovering an
injected direction may have recovered only what we added, and that training on
natural text narrows but does not close that. So: rebuild the readout-22 state as
the *clean* carrier state plus the same `α·u_L` delta, with no forward
computation between, and probe that.

| inject layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| real target | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 |
| synthetic, no forward | 0.167 | 0.167 | 0.167 | 0.167 | 0.333 | 1.000 |
| delta alone | 0.125 | 0.250 | 0.250 | 0.250 | 0.375 | 1.000 |

The natural-text boundary does not recognize the injected vector. The blocks
between the injection site and layer 22 have to transform it first, which is a
claim about the model rather than about our own arithmetic — and the inject-22
column shows exactly what the artifact looks like when it *is* present.
Reproduced by `scripts/analyze_retained.py`.

### Rerun under a repaired control, 2026-08-05

`concepts.random_control` seeded on the bare seed, so all eight concepts got the
same control direction — and since `Intervention` normalizes to unit length, the
per-concept norm was discarded too and the edits were byte-identical. That arm
could not have failed. Fixed, and the confirmatory run repeated into
`results/retained_test_qwen05b_v2_*`.

| layer | 2 | 6 | 10 | 14 | 18 | 22 |
|---|---|---|---|---|---|---|
| target, original | 0.500 | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| target, repaired | 0.500 | 0.193 | 0.198 | 0.125 | 0.130 | 0.141 |
| `random`, original | 0.125 | 0.130 | 0.125 | 0.125 | 0.125 | 0.125 |
| `random`, repaired | 0.125 | 0.115 | 0.141 | 0.115 | 0.125 | 0.115 |

The target arm is bit-identical, as it has to be — control seeding never touched
it. The `random` arm's concept states went from 0.21 apart (fp16 noise) to 54.5
apart, and its accuracy now moves instead of pinning at chance. **The result did
not depend on the broken control**, which is what the rerun was for.

### Threats checked

- **Format collapse.** Strong injections damage formatting. At L2 only 44% of
  target trials put a label as the unrestricted argmax — but restricted to those
  trials accuracy is still **0.435** against 0.125, so the effect is not an
  artifact of the model losing the output format.
- **Damage matching.** At L2 the carrier KL was 1.50 (target), 1.00 (random),
  1.37 (shuffled), with format integrity 0.44/0.21/0.33. This is *approximately*
  matched, not exactly: the target arm carries ~50% more KL than `random`, though
  `random` degrades formatting more (0.21 vs 0.44). `shuffled` is the
  best-matched control on KL (1.37 vs 1.50) and the target still beats it 0.500
  to 0.125. So the L2 contrast is not explained by the target edit being gentler
  — but the residual KL gap is a real limitation, and a cleaner run would
  calibrate each arm's strength to a common KL band on the development split
  rather than matching norms and accepting whatever KL follows.
- **One-concept artifact.** 6 of 8 held-out concepts exceed twice chance at L2
  (`coffee` 1.000 down to `airport` 0.167). The pooled mean is not two concepts
  carrying six.
- **Label position.** Codebook display order is permuted independently of label
  assignment, and `clean` sits at exactly chance, so neither line position nor
  label identity produces above-chance answers on its own.

### Scale extension — exploratory, not confirmatory

`alpha = 1.0` was frozen on 0.5B development concepts and is transferred to other
model sizes **unchanged**, without per-model recalibration. Those runs are
therefore exploratory and must not be quoted as confirmatory, and the layer sets
are matched on relative depth (~8%, 25%, 42%, 58%, 75%, 92%) rather than absolute
index.

`Qwen2.5-1.5B-Instruct` (28 layers), held-out concepts, plain-text ceiling 0.979,
`shuffled` 0.104–0.177 at every site:

| inject layer | 2 | 7 | 12 | 16 | 21 | 26 |
|---|---|---|---|---|---|---|
| target | 0.792 | 0.333 | 0.240 | 0.141 | 0.078 | 0.130 |

Same shape as 0.5B, uniformly higher at the early sites, and still at chance past
the midpoint — so a 3× parameter increase raises the ceiling of the effect
without moving the depth at which it dies.

**Format check passes.** Format integrity at 1.5B layer 2 is very low (0.08), but
restricted to format-intact trials the target arm is still **0.562** against
chance 0.125. The effect is not an artifact of the model losing the output
format. At 3B, format integrity at the early site is **1.00**, so the concern
disappears entirely with scale.

**Damage matching fails at 1.5B layer 2, and this cell should not be quoted.**
Carrier KL is 1.18 for the target arm against 0.38 (`random`) and 0.39
(`shuffled`). The concept direction is aligned with directions the model actually
uses, so at equal *norm* it perturbs behaviour far more than a random direction
does. The +0.615 contrast there is therefore confounded with damage and is not a
clean estimate. This is precisely why the ladder is exploratory: fixing it
requires per-model strength recalibration to a common KL band, not a transferred
alpha. The 0.5B confirmatory run does not have this problem (1.50 target vs 1.00
and 1.37) — the controls there are damaged comparably or more.

**Correction to an earlier reading of layer 21.** I first described the 0.078 at
layer 21 as "not a real anti-effect" by eyeballing it against chance. Against the
*strongest control* the cluster bootstrap gives −0.068 [−0.104, −0.031], which
excludes zero. It is a genuine below-control dip in this run. It is still one
cell of an exploratory, damage-unmatched sweep, so the right treatment is to
report it and not interpret it, rather than to dismiss it.

`Qwen2.5-3B-Instruct` (36 layers), held-out concepts, ceiling 0.938,
`shuffled` 0.120–0.172 at every site:

| inject layer | 3 | 9 | 15 | 21 | 27 | 33 |
|---|---|---|---|---|---|---|
| target | 0.823 | 0.677 | 0.333 | 0.354 | 0.104 | 0.104 |
| effect vs strongest control | +0.661 | +0.505 | +0.208 | +0.224 | −0.026 | −0.021 |

**3B is the cleanest run of the three, and the reason is worth stating.** Format
integrity is **1.00 at every site** — the model never stops emitting a label — so
the format caveat that complicates 0.5B and 1.5B does not apply at all. At layer
3 all **8 of 8** concepts clear twice chance.

**And the damage confound inverts at mid-depth, which is the strongest single
result here.** At layer 21 the target arm's carrier KL is 1.22 while `random` is
1.83 and `shuffled` is 2.91 — both controls disturb the model *more* than the
real concept does. They still sit at 0.125 and 0.130 while the target reaches
0.354, giving +0.224 [+0.135, +0.323]. Layer 15 is the same story (`shuffled` at
KL 1.91 versus target 1.22, and still at chance). Those two cells cannot be
explained by "the concept edit was simply more disruptive", which is the standing
objection to the early-layer cells.

The early 3B cells do **not** get that defence: at layer 3 the target arm carries
KL 2.44 against 0.51 and 0.41 for the controls. That is a worse mismatch than
1.5B's, and layer 3 should be read with the same caution.

### What the ladder shows

| depth | 0.5B | 1.5B | 3B |
|---|---|---|---|
| ~8% | 0.500 | 0.792 | 0.823 |
| ~25% | 0.193 | 0.333 | 0.677 |
| ~42% | 0.198 | 0.240 | 0.333 |
| ~58% | 0.125 | 0.141 | 0.354 |
| ~75% | 0.130 | 0.078 | 0.104 |
| ~92% | 0.141 | 0.130 | 0.104 |

![usable depth by scale](../figures/retained_scale.png)

The depth at which the channel closes **moves later with scale**: 3B is still at
0.354 where both smaller models are at chance. All three are dead by ~75% depth.
An earlier draft of this note said scale raises the ceiling "without moving the
depth at which it dies" — that was written from the 1.5B data alone and 3B
contradicts it. Three points is not a scaling law, but the direction is
consistent, and it is the direction that would reconcile this small-model result
with Lindsey's report of frontier introspection peaking around two-thirds depth.

### Known quirk in the raw schema

`carrier_kl` is meaningless for the `natural` rows. That arm deliberately uses a
*different* carrier — the concept is prepended in plain text — so its KL is
computed against a clean cache built from a different token sequence. No analysis
reads it: the damage table covers only `target`/`random`/`shuffled`, and the
usability table's KL column is target-only.

I am leaving the field as generated rather than patching it and re-running. The
confirmatory split has been consumed; re-running it to tidy an unused column
would mean the published numbers came from code that never produced them, which
is a worse trade than documenting the quirk.

### What this does and does not license

It licenses: *causal use of a retained activation trace, at early injection
sites, in this model, through this interface*. It does **not** license
introspection, self-knowledge, privileged access, or metacognition. No
self-report-specific control has been run, and a generic state-conditioned
classifier would show the same thing.

It is also, per the prior-art section above, **a replication**: the schedule is
Lindsey's, and the early-layer-only profile is already reported for
Llama-3.1-8B. The contribution is the answer space and the scale, not the
phenomenon.
