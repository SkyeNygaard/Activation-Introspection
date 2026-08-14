# Application strategy

Written **2026-08-14**, from a working session that reviewed the repository
against what the SPAR mentors and the MATS 12.0 guidance actually ask for. This
is judgement and argument, not measurement. Every number in it is sourced from
[`CLAIMS.md`](CLAIMS.md) or the notes; nothing here is new evidence.

It exists because the reasoning would otherwise be lost, and because the same
questions ("should I switch model?", "is any of this novel?") keep being asked and
re-answered from scratch.

---

## 1. The diagnosis

**The repository reads as a sequence of failures. It is not. It is one claim, told
as six disappointments.**

Every study here reaches the same place from a different direction: whenever an
outside method is handed the same internal state at the same cost, it wins. The
model's advantage exists only where the comparator is denied access. That is a
finding about *measurement*, and it is the thing the work has actually
established.

The failures are the evidence for it. Told as a chain of things that didn't work,
it reads as defeat. Told as "the field's positive results may be artifacts of an
unfair comparison, here is a clean instrument that shows it", the same six
retractions become the argument's credibility.

**Do not lead with a new experiment. Lead with the instrument and the honesty.**

---

## 2. What each mentor asked, and what already answers it

### Noah Siegel — Faithfulness, Self-Knowledge, Introspection

His question: does a self-report reflect *privileged self-knowledge* rather than
generic state-conditioned computation, surface leakage, or perturbation damage?

**Already answered, negatively, with the field's own criterion.** On the identical
576 episodes behind the 0.891, a four-shot nearest-centroid reader given the same
post-injection states scores **1.000** against the model's **0.892**, with **62
reader-only episodes and zero model-only**. Under the operative definition in
[2508.14802](https://arxiv.org/abs/2508.14802) there is no privileged access at
this site.

The second thing for him: **the model can use what it cannot say.** Its forced
choices carry real information about its internal state; its free descriptions
carry none — leave-one-carrier-out classifiers on the report text score 0/24, and
it says `hammer` on 14 of 24 reports regardless of what was injected. That is the
faithfulness gap in a controlled setting.

### Belinda Li — Introspection Training for Verbalization of Activations

Her question: can training make verbalizations track injected content **on
held-out concepts**, rather than prompt cues or generic perturbation?

**The phrase "held-out concepts" is doing hidden work, and that is the
contribution.** Two readings, opposite answers:

| sense of "held out" | result |
|---|---|
| a **direction** the model never trained on | training works — **0.927** |
| an **exemplar** of a category whose other members were demonstrated | **0.083**, at the floor, while a cheap reader on the identical states gets **0.986** |

If that distinction holds, work reporting the easy sense while implying the hard
one is overclaiming. **Caveat that must travel with it:** those two numbers
currently come from *different setups* — 0.927 from a trained model, 0.083 from an
in-context one. A sceptical reader kills the comparison immediately and would be
right. Putting both on one trained checkpoint is the single highest-value run
available and it has not been done.

### The paragraph that is a gift to Li specifically

Version 1 of the trained reporter scored **0.917 while holding 5×10⁻⁹ probability
on the answer tokens**. The loss was a two-way softmax over the label logits, so
the adapter learned to suppress both labels and keep the right one on top — a
probe wearing the model's output head, and **no forced-choice metric can detect
it**. Her project proposes to train exactly this kind of thing. This is a worked
account of a trap sitting in its path, found by falling in and climbing out. It
does not depend on model size and would happen at any scale.

It is currently filed under "two failed versions are the more useful half" in
[`PROJECT-BRIEFS.md`](PROJECT-BRIEFS.md). It deserves to be prominent.

---

## 3. MATS 12.0 (Neel Nanda) — what the guidance actually says

Read from the application document on 2026-08-14. Deadline **Fri 4 September**.

**Reuse is allowed and lands in the good bucket.** His rule: if the work was done
on your own, in ≤20 hours, and *not for* the application, "this is obviously fine,
and you can just treat it as a normal application project." The harsher-judging
warning applies to people submitting a prior paper *instead of* doing a project,
which is a different case. Skye states the work is under 20 hours. Include an
hours estimate; he suggests tracking with Toggl. Not counted: waiting for training
to finish, general prep, the application form itself. Counted: coding, analysis,
planning, papers read *for* the project, and the write-up.

**"Nothing novel to say" is the wrong worry.** His common-mistakes list reads like
a description of this repo, in the good direction:

- "negative results are fine! Lying about them is not"
- "A really *positive* sign is when I think of a way the results could be false,
  then discover you've already checked it"
- "Skipping the cheap control: replace your vector with a random one, compare
  against 'just ask the model'" — both done here
- "Building on a phenomenon without first checking it replicates in your setting"

**Introspection is on his list.** In the model-diffing section: *"I really liked
the idea of introspection adapters. Lots of room to do better, and to find real
use cases."* And his stated shift is toward interpretability that does something
useful, **measured against baselines** — which is this project's whole method.

**The real weakness is model choice, not novelty.** "Only studying old models
(GPT-2, Pythia, Gemma 2)" and "working with a model that's just way too dumb for
the task" are both on the list.

**But that objection has an answer that is not a model swap.** The anchor
condition — state the concept in plain text, everything else identical — scores
**0.875**. The model can do the task when the information is visible. So the 0.083
held-out failure is a real finding, not a weak model. Point at that number.

**What he is not excited by:** SAE hill-climbing, circuit finding for its own
sake, toy models, very theoretical work. A pipeline of "SAE feature diff →
transcoder → circuit localization" aims at the part of interpretability he has
moved away from. Even inside model diffing he writes that black-box diffing agents
"work surprisingly well, I'd recommend starting here."

---

## 4. Decisions taken, with reasons, so they are not relitigated

### Do not switch models

Considered and rejected on 2026-08-14. Reasons, in order of weight:

1. **The "too dumb" objection is already answered** by the 0.875 anchor.
2. **Switching costs everything.** Every frozen protocol, control and note is on
   Qwen2.5-3B. The alternative is redoing it all or holding results on two models
   that cannot be compared, which is worse than either.
3. **The findings that carry the application are not about this model's
   abilities.** The training-loss trap is scale-independent; the comparator result
   is about what the baseline is given; the "held-out" point is definitional.

**What was done instead:** `Qwen3-4B-Instruct-2507` is used for new inference-only
work (see notes/38). It is a current-generation model with the *same 36-block
layout* as Qwen2.5-3B, so injection layers transfer directly and depth results stay
comparable. It fits for inference (~10.7 GiB) but **not for training** (~15 GiB).

Rejected alternatives and why:

- **Qwen3.5 small models** (0.8/2/4/9B, released 2026-03-02): 18 of 24 layers are
  Gated DeltaNet linear attention, so "layer 9" is not comparable to layer 9 in a
  standard transformer, which breaks every depth result here. Kernels are
  CUDA-oriented and may not run on Apple Silicon at all.
- **Gemma 4** (2026-04-02): "E" sizes are *effective* parameters; real footprint is
  larger than the name suggests. No released feature dictionaries as of this date.
- **Gemma 3** has the richest SAE ecosystem (Gemma Scope 2). Rejected because this
  project has no feature-discovery step to use it for — see below.
- **Quantization** to fit something bigger: the standard 4-bit path is CUDA-only;
  MLX quantizes well on this hardware but would mean porting the hooks, the
  interventions and the training off PyTorch/peft. And it changes the activations,
  which confounds the one variable a replication is trying to isolate.

### Keep raw PyTorch hooks; do not migrate to TransformerLens

The general argument for libraries is correct — an agent writes most of the code,
and less custom code means fewer quiet errors. It does not apply *here* because:

- **25 frozen protocols hash `pyproject.toml`** and 55 artifacts hash the source
  modules. Adding a dependency invalidates them and fails two tests immediately.
- **The code already exists and is tested** for exactly the failure modes a library
  would guard: seven hook tests including `test_capture_order_decides_what_is_recorded`,
  which pins the trap where capturing before intervening silently records clean
  activations under an edited label.
- **The port is the risky operation**, not the status quo. The argument applies to
  code not yet written.
- **Libraries catch implementation errors, not conceptual ones.** The three worst
  traps found here — hook ordering, the attention sink at residual norm 1537, and
  needing to centre the concept bank — were all conceptual.

**Choose the library at the start of a project, not in the middle.** To try one
anyway, install into a throwaway environment and leave the frozen pipeline alone.

### The SAE gap is a real limitation — state it, don't fix it

The project cannot say *which features* a trained reporter uses, because its
directions are **constructed** (difference-in-means over contrast prompts) rather
than **discovered**. Related: the injected direction and the cheap reader are close
cousins, which is why notes/13 found the fitted reader *is* the average concept
direction at cosine 0.99999. That is the strongest objection to the comparator
headline and it should be foregrounded rather than left to be found.

---

## 5. What to do, in order

1. **Fix the "held-out" comparison.** Put both senses on one trained checkpoint —
   held-out direction, held-out exemplar, and the cheap reader on both sets of
   identical states. One figure, one run. It removes the objection that the two
   numbers came from different setups.
2. **Promote the training-loss trap** out of a footnote.
3. **Reframe the write-up** around the measurement claim rather than the
   experiment chain.
4. **Optionally**, replicate the in-context result on Qwen3-4B — an hour of
   inference, and it converts "your model is old" into "I checked".

## 6. What not to do

- **No more notes/29–37 prompt descendants.** Two independent outside reviews and
  the handoff agree. The chain narrows its own predecessor at every step.
- **No SAE/transcoder/circuit pipeline** for the MATS application — it targets what
  Nanda says he is least excited by.
- **No model migration** part-way through.
- **Do not hype.** He says explicitly that he can tell, and that negative results
  are fine while dressing them up is not.
