# Pre-run note: the cost criterion has a knob it does not name

Written **2026-08-12, before the run.**

## The objection this run exists to answer

Everything in [`11`](11-matched-cost-reader.md) and
[`15`](15-matched-reader-on-content.md) shows a cheap reader beating the model.
The strongest objection to all of it:

> Of course it does. The reader sees the state at the injection site at full
> strength. The model must propagate that state through 27 further blocks, attend
> to it, combine it with a codebook, and commit to a token. Post-processing cannot
> add information. This is an information-processing inequality, not a fact about
> introspection.

If that is what those runs measured, **1714 of 1728 is a dressed-up restatement of
something true in advance**, and this repository's headline is much weaker than it
reads.

## Where the objection is incomplete, and what that implies

The reader in those designs receives **four labelled examples of the query's own
direction**. It is handed the answer key for the specific question being asked. It
cannot fail, and its 1.000 is close to arithmetic.

The criterion the field uses ([Song et al.](https://arxiv.org/abs/2508.14802))
requires beating a third party of *equal or lower computational cost*. **It says
nothing about supervision.** But supervision is what decides the outcome:

| third party gets | consequence |
|---|---|
| labelled examples of the queried concept | wins trivially; no model could pass |
| no labels at all | has nothing to fit, while the model has its pretrained semantics |

So the verdict flips on a resource the criterion does not price. **A test no system
can pass is not measuring a capability.** That is a defect in the definition, it is
the reason my own results may be worth less than they look, and it is checkable.

## What I am about to do

Test the regime the criterion should care about and my designs have never run:
**identify the injected concept with no demonstrations at all.**

Inject one of eight concepts at the usual site, then ask which of those eight
words it was. No worked examples. Chance is 0.125.

| arm | what it is | why |
|---|---|---|
| `model_injected` | model asked which concept was added | the question |
| `lens_injected` | the same state read through the final norm and unembedding, scored over the same eight words | **a label-free reader, and cheaper than the model by 27 blocks** |
| `model_visible` | the word written in plain text, same question | **capacity ceiling.** Without this a null is uninterpretable |
| `model_clean` | nothing injected | the model's standing bias over the eight words |
| `lens_clean` | lens on the unedited state | the lens's standing bias |

Strengths 1, 2 and 4, because an edit tuned for a two-way sign decision may be far
too weak to name.

## What each outcome means

| Outcome | Reading |
|---|---|
| `model_injected` above chance **and above** `lens_injected` | **The first privileged-access positive in this repository.** The model would be doing something a cheap label-free reader cannot, in exactly the regime the criterion was written for |
| `lens_injected` well above `model_injected` | The direct read wins even without labels. That does **not** rescue the criterion — it strengthens the argument that it is close to unsatisfiable — but it removes the supervision knob as the explanation |
| Both at chance, `model_visible` high | The model cannot name its own injected states at 3B. A clean capability limit, and it bounds what the 0.899 in [`14`](14-content-versus-disturbance.md) may be called: discriminating two options is not identifying one |
| `model_visible` at chance | The model cannot do eight-way selection at all. Harness or task fault; read nothing into the other arms |

The third outcome is the one I consider most likely, and it is unflattering to the
content result. Saying so now.

## What it costs

Eight concepts × three carriers × three strengths, plus the control arms. A few
hundred forward passes, inference only, no training.

---

# Result: the supervision hypothesis is wrong, and the criterion is worse off than I thought

Run **2026-08-12**. 39 seconds. Artifacts: `results/zero_shot_identify_v1_raw.jsonl`,
`results/zero_shot_identify_v1_summary.json`. Runner:
`scripts/run_zero_shot_identify.py`.

| arm | accuracy | n |
|---|---:|---:|
| `model_visible` — capacity ceiling | **1.000** | 24 |
| `lens_injected` — label-free, one matmul | **0.986** | 72 |
| `model_injected` — the model naming its own state | **0.597** | 72 |
| `model_clean` — standing bias | 0.000 | 3 |
| `lens_clean` — standing bias | 0.000 | 3 |

Chance is **0.125**.

| injection strength | model | lens |
|---:|---:|---:|
| 1.0 | 0.458 | 0.958 |
| 2.0 | 0.667 | 1.000 |
| 4.0 | 0.667 | 1.000 |

## Two things, and the first is good news

**The model can identify its own injected concept.** 0.597 against a chance of
0.125 — nearly five times chance, on an **eight-way** naming task with no
demonstrations. Every earlier result in this repository is two-way discrimination;
this is the first identification result here, and it is a real capability. Both
clean arms sit at 0.000, so it is not a standing preference for some option.

**And it still loses, badly, to one matrix multiply.** Reading the same state
through the model's own final norm and unembedding — no labels, no demonstrations,
no examples of anything — gives **0.986**.

## My hypothesis was wrong

The pre-run argument said the criterion's verdict turns on a knob it does not
price: give the third party labelled examples and it wins trivially, withhold them
and the model's pretrained semantics should give it the edge.

**Removing the labels changed nothing.** The label-free reader beats the model by
0.389. Supervision is not what was carrying the earlier results, and the
explanation I proposed is dead.

## What replaces it is stronger, and less comfortable

The objection I opened this note with — that post-processing cannot add
information, so a direct read must win — is **correct**, and this run is the
cleanest demonstration of it in the repository. The model's answer is a lossy
function of a state that the reader gets intact.

But it sharpens into something the criterion's authors would have to answer:

> A model asked about its own internal state is competing **against its own output
> head, applied directly to that state**. The unembedding is part of the model, it
> is vastly cheaper than 27 further blocks, and it is available to any third party
> with activation access. Wherever a state is linearly decodable, the
> equal-or-lower-cost criterion cannot be satisfied — not because the model lacks
> introspective access, but because the comparison class contains a shortcut
> through the model itself.

And there is a specific reason that bites this whole literature. **Concept
injection plants vectors built from token contrasts.** They are linearly decodable
by construction — that is what makes them concept vectors. So every
concept-injection paradigm is testing introspection precisely where the criterion
is unsatisfiable, and a negative verdict there carries much less than it appears
to.

**The design principle that follows:** the criterion is only informative on states
a lens *cannot* read. That is a testable precondition, it costs one matmul to
check, and no run in this repository has ever checked it.

## Epistemic status

- **Observed:** the five numbers above, on one model, one site, three strengths.
- **Inferred:** that the criterion is unsatisfiable wherever states are linearly
  decodable. This follows from the measurement plus the data-processing argument,
  but it is an argument, not a measurement.
- **Speculated:** that this generalises to the concept-injection literature. The
  reasoning is that those vectors are token-contrast constructions, which is true
  of the ones built here and stated of others, but I have not verified it for any
  other group's vectors.

## Limits

- One model, one site, one bank of eight concepts, three carriers.
- The lens uses the model's own unembedding, so calling it a "third party" is
  generous — that is the point being made, not an oversight, but it should be
  stated whenever the number is quoted.
- 0.597 for the model is a real capability and a **weak** one. It is not evidence
  that the model has useful self-knowledge, only that it has more than chance.
- This says nothing about states the model computed itself, which remain the
  regime where the criterion could still be informative — and where, by this
  note's own argument, it should be tested.

## Novelty, before the result exists

The supervision gap in the criterion is the conceptual claim, and I have **not**
searched for it. [Detecting the Disturbance](https://arxiv.org/abs/2512.12411) and
[Content-Agnostic](https://arxiv.org/pdf/2603.05414) critique the introspection
literature but on different grounds. Whether anyone has argued that the
cost criterion is underdetermined with respect to the comparison method's
supervision is **unchecked**, and must be searched before the claim is made
anywhere that matters.
