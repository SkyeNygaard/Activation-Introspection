# Handoff

Written **2026-08-12**. For a person or agent picking this up with no prior context.
Read this first, then [`spar-application/RESEARCH-DIRECTION.md`](spar-application/RESEARCH-DIRECTION.md),
then the notes in order.

---

## 1. What this project is

Skye is applying to SPAR (Fall 2026) as an ML engineer moving into empirical AI
safety. Six applications; **original research for two of them only**:

| # | Project | Mentor | Effort |
|---|---|---|---|
| 1 | Introspection Training for Verbalization Activations | Belinda Li | **research** |
| 3 | Faithfulness, Self-Knowledge, and Introspection | Noah Siegel | **research** |
| 2, 4, 5, 6 | Programmatic Attention; In-the-Wild Control; Reward-Seeking; CoT Obfuscation | — | application only |

The research question both share:

> **Can a model tell us what is happening inside it, and how would we know if it
> could?**

**Standing constraints.** No further LoRA training (decided; the training arm is
complete, not abandoned). Local models ≤3B on an M4 Pro with 24 GB shared memory.
Everything since 2026-08-12 is inference-only.

---

## 2. Where things stand — the one-paragraph version

A model reading its own injected internal state loses to a trivial outside method
whenever that method is given the activations, and wins whenever it is not. That
turns out to be the whole story: **"privileged access" is a step function in what
the comparator is handed, not a property the model has or lacks.** The model sits
strictly in the middle and never moves. Separately, and more usefully: how much a
model can *say* about its own state varies 2.4× on prompt wording alone, and the
intuitive prompt ("name the concept") is among the worst.

**Added 2026-08-12 (notes 22–24), and it sharpens the above.** The one apparent
exception to the dominance relation — 14 episodes where the model beat the reader
at weak strength — was a scoring artifact and does not exist. And what looked
like the model recognising *content* is recognising a *vector it was already
shown*: move the query exemplar out of the demonstrations and the model drops to
0.083 on twin pairs, below the 0.25 coin-flip null, while the cheap reader on the
identical states holds at 0.986. Five instruction wordings do not move it. So the
model is not merely worse than the outside method — on a task requiring
generalization it is at the floor, using none of information that is provably
present.

---

## 3. The results that hold, with their numbers

Full detail in [`spar-application/CLAIMS.md`](spar-application/CLAIMS.md), which
grades every statement including retracted ones.

**The comparator ladder** ([notes/20](activation-introspection/notes/20-comparator-tiers.md)).
Eight-way identification of an injected concept, chance 0.125:

| comparator gets | accuracy |
|---|---:|
| prompt only | 0.125 *(by construction)* |
| the model's own words, no activations | 0.292 |
| *the model itself* | *0.667* |
| activations (logit lens) | 0.986 |

**The cost criterion across four task shapes** ([notes/15](activation-introspection/notes/15-matched-reader-on-content.md)).
Model vs a four-shot nearest-centroid reader refitted per episode:

| task | model | reader | model-only episodes |
|---|---:|---:|---:|
| polarity (`+v` vs `−v`) | 0.917 | 1.000 | 0 |
| content (`v_A` vs `v_B`) | 0.899 | 1.000 | 0 |
| random directions | 0.663 | 1.000 | 0 |
| weak nudge (strength 0.15) | 0.497 | 0.833 | 14 |

**Those 14 episodes are gone.** [notes/22](activation-introspection/notes/22-the-weak-arm-was-a-floor-not-a-frontier.md)
rescored the same saved rows at the protocol's own unit — the twin pair, where a
cell counts only if both byte-identical members get their opposite labels right —
and 14 model-only episodes become **1 model-only pair**. At strength 0.15 the
model scores 1 pair in 144 against a 0.25 null, because in 16 of 24 cells it emits
one constant label whatever was injected. Row accuracy averages 0.4965, which
reads as chance and means blind; notes/08 had already recorded that floor. **There
is no regime where the model is ahead.** Quote the row-level table above only with
this correction attached.

**Held-out generalization: the model is at the floor**
([notes/23](activation-introspection/notes/23-held-out-semantic-generalization.md),
[notes/24](activation-introspection/notes/24-is-the-held-out-failure-the-interface.md)).
Give every injection position a different exemplar and hold the query exemplar out
of the demonstrations. Twin-pair accuracy, null 0.25:

| arm | model | reader |
|---|---:|---:|
| same exemplar (anchor) | 0.521 | 1.000 |
| held-out, real categories | **0.083** | 0.986 |
| held-out, arbitrary groupings | 0.076 | 0.333 |

The model gains nothing from the categories being real; the reader gains
everything. A geometry gate run first showed layer 9 carries the category cleanly
(held-out nearest-centroid 1.000 and 0.989 on the two frozen pairs), so this is
not a capacity failure. Five instruction wordings, development and confirmation
pairs split before the run: **no cell of ten beats the null**, pooled held-out is
60/360 = 0.167 against a pooled anchor of 0.681. The wording is not inert —
telling the model to attend to its own state cuts constant-labelling from 40% to
25% and lifts the anchor to 0.875 — which is what makes the null decisive.

**The model reads meaning, not just disturbance**
([notes/14](activation-introspection/notes/14-content-versus-disturbance.md)).
Two different concepts discriminated at 0.899; two random directions at **identical
separation by construction** at 0.594; 4 of 4 pairs.

**Elicitation dominates** ([notes/21](activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md)).
Same state, same reader, wording only: `sensory` 0.708, `associations` 0.500,
`name_one` **0.292** (tied with a prompt that forbade naming).

**Clustering predicts learnability** ([notes/16](activation-introspection/notes/16-visible-rule-capacity.md),
[notes/19](activation-introspection/notes/19-clustering-predicts-learnability.md)).
Whether the four-shot interface can learn a hidden rule is predicted by whether the
rule's classes clump in representation space. **12 of 14 predictions correct with
thresholds frozen to disk before any accuracy existed** — the only prospective test
in the repository. Positive clumping meant learnable **8 for 8**; negative meant not
learnable 4 of 6. Use it as a green light, not a veto.

**Calibration is unusable** ([notes/20](activation-introspection/notes/20-comparator-tiers.md)).
Confidence 0.998 when right, 0.928 when wrong. A 0.07 gap across a 100% accuracy
gap, so confidence cannot filter self-report.

**Training** ([notes/07](activation-introspection/notes/07-trained-activation-reporter.md),
[notes/08](activation-introspection/notes/08-sensitivity-specificity-tradeoff.md)).
Extends the detection floor to nudges the base model is blind to (0.790–0.863 where
base is 0.500) and destroys selectivity (random directions 0.513 → 0.913–0.955).
**The adapters were never saved**, so none of this can be re-scored. That is why
later notes compare against published numbers.

---

## 4. What has been retracted — read this before trusting any older text

Six corrections, all self-inflicted, all left visible above their fixes in the notes.

| claim | what killed it |
|---|---|
| The 0.891 shows introspection | A four-shot reader gets 1.000 on the same 576 episodes ([11](activation-introspection/notes/11-matched-cost-reader.md)) |
| These numbers measure *which concept* | The bank shares one axis; the fitted reader **is** the average concept direction, 0.99999 ([13](activation-introspection/notes/13-shared-axis-audit.md)) |
| Training is probe distillation and loses | True only where the probe was fitted to be optimal ([15](activation-introspection/notes/15-matched-reader-on-content.md)) |
| Training buys generality probing cannot have | The probe was handicapped; a per-episode reader gets 1.000 off-axis ([15](activation-introspection/notes/15-matched-reader-on-content.md)) |
| "28 of 28 pairs positive" | `abs()` was applied before counting. True when measured properly, but that measurement never showed it ([13](activation-introspection/notes/13-shared-axis-audit.md)) |
| The model knows 5× more than it says | The elicitation prompt forbade naming. Gap closes entirely with a better prompt ([21](activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md)) |

**The pattern, named so it is not repeated:** a measurement artifact gets promoted
to a finding about the model, and the correction always comes from testing the
boring explanation that was not ruled out. Twice the second error was made *while
fixing the first*.

---

## 5. Novelty — most of this is not new

[`spar-application/LITERATURE-BOUNDARY.md`](spar-application/LITERATURE-BOUNDARY.md)
has the full accounting. Two searches, both run **after** the results existed, in
violation of that file's own rule.

**Four of six candidates are prior art:**

- Training loses to a probe → [Looking in the Mirror](https://arxiv.org/html/2608.04347) (5 Aug 2026)
- Shared axis / concept-agnostic machinery → [Mechanisms of Introspective Awareness](https://arxiv.org/html/2603.21396v1)
- "Injection site is trivially decodable" → an already-stated criticism
- "Used but lens-illegible" → [Steerable but Not Decodable](https://arxiv.org/html/2604.02608v2)

**Nothing is both unclaimed and standing.** As of 2026-08-12:

- The clustering→learnability result was the only unclaimed candidate, and
  [26](activation-introspection/notes/26-someone-elses-rules.md) **failed to
  replicate it** on rules written by a model blind to the hypothesis. Rank
  agreement 0.403 against 0.785, permutation p = 0.078, and the usable half — a
  green light at 8 for 8 — fell to 4 of 7. Do not put it in an application.
- The elicitation range ([21](activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md))
  is still unclaimed and still standing. One model, one setup, so it is an
  observation rather than a law — but nobody appears to have written it down, and
  it bears directly on Belinda Li's question about verbalization without training.
- The sensitivity/specificity trade-off ([08](activation-introspection/notes/08-sensitivity-specificity-tradeoff.md))
  had **never been novelty-checked at all** and now has been: three of its four
  parts are prior art, and the conjunction is an extension. See
  [`PAPERS-REVIEWED.md`](spar-application/PAPERS-REVIEWED.md), which is the new
  ledger of what has actually been read and to what depth.

**There is now one candidate, added the same day**
([29](activation-introspection/notes/29-can-abstention-recover-selectivity.md)),
and it is the strongest thing here:

> **Letting an introspective monitor abstain does not repair it — and in a trained
> reporter it makes things worse.** Dropping the least-confident self-reports
> narrows the gap between real concepts and meaningless directions (0.059 → 0.013
> and 0.099 → 0.019) while nearly *doubling* it in the untrained model (0.232 →
> 0.455). The trained reporter is most confident precisely where it is wrong.

Why this one is different from everything above: selective prediction is a mature
tool that **has never been pointed at a model's reports about its own internals**;
it answers a fix Anthropic explicitly names as unbuilt; and it corrects this
repository's own [20](activation-introspection/notes/20-comparator-tiers.md),
whose "calibration is unusable" was an artifact of averaging a ceiling-squashed
number instead of ranking it. It cost no GPU — the rows were already on disk.

**It is still an extension, not a discovery.** One model, one recipe, one strength
cell, a secondary analysis of artifacts frozen for another purpose, and it does not
test the DPO-refined adapters Anthropic actually proposes. Say that alongside it.

**Resolved 2026-08-12.** [Emergent Introspection in AI is Content-Agnostic](https://arxiv.org/pdf/2603.05414)
(Lederman and Mahowald) has now been read in full. It is **not** in tension with
[notes/14](activation-introspection/notes/14-content-versus-disturbance.md) —
they score open-ended naming of the injected concept, notes/14 scores forced
choice between two arbitrary labels, and a model can fail the first while passing
the second. But it does publish the conclusion
[notes/23](activation-introspection/notes/23-held-out-semantic-generalization.md)
reached, from confabulation statistics rather than causally (74.8% of one model's
wrong guesses are the word "apple"). **Notes 23–24 are independent convergence by
a stronger method, not a new claim.** The method is the claimable part. Full
accounting in [`LITERATURE-BOUNDARY.md`](spar-application/LITERATURE-BOUNDARY.md).

[Li et al. 2511.08579](https://arxiv.org/abs/2511.08579) — the top-ranked
mentor's own paper, and the target of that application's critique question — has
also now been read in full and is recorded in the same file. **No draft critique
text exists anywhere in this repository, deliberately:** that writing is under an
attestation and must be Skye's own.

---

## 6. What is blocked, and the one branch that is open

**Natural states — reporting on something the model computed itself — is the
biggest gap.** Five runs failed. The blocker is now known precisely: not the
transplant (works, 9 of 12 pairs certified), not the interface (works), but the
hidden rule. Parity scores 0.533 *with the arithmetic in plain sight*, because its
classes are **anti-clustered** (−0.023) — each expression sits closer to the
opposite class than its own.

**It is reopenable**, with conditions, all from
[notes/16](activation-introspection/notes/16-visible-rule-capacity.md):

1. Pick a hidden class whose members **clump** (semantic categories reach 0.885).
2. Freeze it before seeing any natural-state data — `category` was chosen by looking
   at results, so it is a lead, not a finding.
3. Use a **third** bank; the two earlier ones are spent.
4. Keep the nothing-hidden screen as a gate so a null reads as a real null.
5. Certify each transplant individually.
6. **New:** check the class clumps *and* that a lens cannot read it, before spending
   anything. Both checks cost ~200 forward passes.

---

## 7. What I would do next, in order

**Items 1 and 2 below are done as of 2026-08-12.** Both are struck through rather
than deleted, so the ordering that produced the current state stays visible.

1. ~~**Read [2603.05414](https://arxiv.org/pdf/2603.05414) in full.**~~ Done. See
   §5 — no contradiction, but it publishes notes/23's conclusion first.
2. ~~**Widen the elicitation search.**~~ Done, [notes/24](activation-introspection/notes/24-is-the-held-out-failure-the-interface.md).
   Five wordings on the held-out task, dev/confirmation split before the run. Not
   one cell of ten beats chance, and the kill rule fired: **stop varying
   elicitation for this interface.** This also freezes the
   elicitation-optimised baseline any training study needs — wording buys zero on
   held-out, so anything training adds is training's.
3. ~~**Test the clustering gate on a rule set written by someone else.**~~ Done,
   and it **failed**: [notes/26](activation-introspection/notes/26-someone-elses-rules.md).
   Fourteen rules written by a model blind to the hypothesis, thresholds imported
   rather than refitted. Rank agreement 0.403 against 0.785, permutation p = 0.078,
   and the usable half — a green light at 8 for 8 — fell to 4 of 7. That closed the
   only unclaimed candidate.
4. ~~**Chain of thought as the last escape hatch on notes/23–24.**~~ Done, and it
   **did not answer the question**:
   [notes/25](activation-introspection/notes/25-does-reasoning-out-loud-rescue-it.md).
   Letting the model reason out loud collapsed the anchor task from 0.694 to about
   0.33, so held-out is uninterpretable. The kill rule was **held, not fired** —
   scoring a broken instrument is not a result. The fix is named there:
   [notes/20](activation-introspection/notes/20-comparator-tiers.md)'s two-stage
   tier never shows the model the letters `Q` and `K` at all.

**Where the live work is now.**

5. **[notes/29](activation-introspection/notes/29-can-abstention-recover-selectivity.md)
   is the strongest thing here and it is finished.** Selective prediction pointed at
   introspective self-reports for the first time. It withdrew notes/20, extended
   notes/08, and answered a fix Anthropic names as unbuilt — negatively. **The
   obvious next step is the one this machine cannot run:** the same measurement on
   the DPO-refined adapters that paper actually proposes, which are trained to
   prefer accurate reports over plausible ones. That is a proposal, not a run.
6. ~~**[notes/30](activation-introspection/notes/30-does-it-know-it-is-about-to-be-wrong.md),
   the natural-states branch redesigned to remove what killed it.**~~ Done, and it
   **closes the branch**. On a state the model computed itself, its prospective
   self-knowledge does not exceed what problem difficulty already explains — **the
   size of the multiplication alone scores 0.819 against the model's 0.805**. A
   post-hoc lead in the hardest third was declared as post-hoc and then killed by
   its own pre-registered confirmation (0.045, CI [−0.065, 0.150]).

   **Read its methodological finding before attempting natural states again.** The
   injected-state work here is interpretable because byte-identical visible text
   pins an input-only learner at exactly 0.500 *by construction*. A natural-state
   design cannot have that control, because the input is what varies and the input
   predicts the outcome. Escaping "everything is planted" costs the thing that made
   the planted results mean anything. That is a harder obstacle than the clumping
   problem in section 6, and it needs an answer **before** the next attempt, not
   after.

**Do not:** run more LoRA; add concept pairs or layers for robustness before the
above; run the criterion against a comparator with activation access and read
anything into it; or quote the clustering gate — it did not replicate.

---

## 8. Conventions that will bite you

- **Write a pre-run note before every experiment** — what, why, what each outcome
  means, what it costs, and a prediction. Every note here has one, with wrong
  predictions left in place. This is Skye's standing instruction, not a nicety.
- **Smoke first**, and disclose the smoke whatever it said.
- **Never edit `src/introspect/hooks.py`** (or other protocol-bound sources) casually.
  Frozen protocols record source hashes and `tests/test_analyze_attention_localization.py`
  fails the moment one moves. A one-line "no-op" fix on 2026-08-12 silently
  invalidated two published artifacts and was reverted; the guard now lives locally
  in `scripts/run_comparator_tiers.py` with the reason in its docstring.
- **Refuse to overwrite artifacts.** Every runner checks and exits.
- **Machine limits.** Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` **and**
  `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0`; `HF_HOME=$PWD/hf_cache`; run
  `introspect.preflight.check` first and let it refuse. **Never two model jobs at
  once** — one was killed mid-load that way.
- **Independent unit is rarely the row.** For random-direction arms it is the
  *direction* (8), not the row (48) — the pattern is all-or-nothing per direction.
- **Plain language in every report.** No jargon without a definition in the same
  sentence. Skye's global instruction.

---

## 9. Repository map

| path | what it is |
|---|---|
| `activation-introspection/notes/01`–`21` | the lab record, in order. Each carries its pre-run reasoning |
| `activation-introspection/scripts/run_*.py` | one runner per experiment; all inference-only since notes/13 |
| `activation-introspection/results/` | protocols, raw rows, manifests, summaries |
| `activation-introspection/src/introspect/` | shared apparatus. **Protocol-bound — see §8** |
| `spar-application/CLAIMS.md` | the claim ledger. The authority on what may be said |
| `spar-application/RESEARCH-DIRECTION.md` | the allocation decision and current direction |
| `spar-application/LITERATURE-BOUNDARY.md` | novelty accounting; where the searches fall short |
| `spar-application/README.md` | the application front page |
| `adaptive-monitor-sandbox/` | second repo: agent + monitor world. Unused since 2026-08-11 |

Git: branch `stage1b-stop-and-trained-reporter`, remote
`github.com/SkyeNygaard/retained-trace-study`. **Another session has committed to
this branch concurrently** — check `git log` before assuming your working tree is
current.

---

## 10. The honest summary for an application

Not *"I trained an introspective reporter."* Closer to:

> I built adversarial controls for activation self-reporting, repeatedly found that
> apparently strong introspection results collapse under better controls — including
> six of my own — and established that the field's central criterion returns
> whichever answer its comparator was set up to give.

The strongest single asset is not a number. It is that every experiment has its
reasoning written beforehand, and the wrong predictions are still sitting next to
the results that corrected them.
