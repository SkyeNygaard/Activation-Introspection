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

**14 episodes in 1728** where the model succeeds and two averages fail. All 14 are
in the weak regime — the only place the model was ever ahead.

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

**Still unclaimed after two searches:** the clustering→learnability result
([16](activation-introspection/notes/16-visible-rule-capacity.md),
[19](activation-introspection/notes/19-clustering-predicts-learnability.md)), and
the elicitation range ([21](activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md)).

**Unresolved and important:** [Emergent Introspection in AI is Content-Agnostic](https://arxiv.org/pdf/2603.05414),
by two authors of the cost-criterion paper this work leans on, is in tension with
[notes/14](activation-introspection/notes/14-content-versus-disturbance.md).
**It has not been read in full and must be before notes/14 is described as novel.**

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

1. **Read [2603.05414](https://arxiv.org/pdf/2603.05414) in full.** It may contradict
   notes/14. Cheapest thing with the largest effect on what can be claimed.
2. **Widen the elicitation search.** [notes/21](activation-introspection/notes/21-is-the-channel-narrow-or-was-i.md)
   found a 2.4× range over six prompts I wrote. A systematic search establishes
   whether 0.708 is a ceiling, and it is the baseline introspection *training* has to
   beat — which no paper appears to establish.
3. **Test the clustering gate on a rule set written by someone else.** The weakest
   link in the only unclaimed result is that I wrote all fourteen rules.
4. **Then** natural states, under section 6's conditions.

**Do not:** run more LoRA; add concept pairs or layers for robustness before the
above; or run the criterion against a comparator with activation access and read
anything into it.

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
