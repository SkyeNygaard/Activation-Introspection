# Pre-run note: I narrowed the channel I then measured

Written **2026-08-12, before the run.**

## A confound of my own making

[`20`](20-comparator-tiers.md) reports that the model reaches 0.667 on its own
injected state while its words transmit only 0.292, and that reports name the
target concept 1 time in 24. I called the verbalization channel the bottleneck.

**The elicitation prompt I used says, verbatim:**

> "Describe what it feels like or what comes to mind, in one sentence.
> **Do not guess a single word; just describe.**"

I instructed the model not to name the concept, then reported that it almost never
named the concept. The 0.042 naming rate is partly my instruction. **That is a
self-inflicted confound and it undercuts the strongest sentence in `20`.**

The 0.292 is still a real measurement of *that* channel, and `20` does call it a
floor rather than a ceiling. But it called it a floor without evidence, and the
obvious first explanation — that I gagged the model — was not tested.

## The question

**How much of the 0.375 gap between reaching and saying is the elicitation, and
how much is the model?**

This matters well beyond fixing my own error. If a better prompt closes most of the
gap, then what looks like a verbalization deficit is prompt design, and
"introspection training" would substantially be learning an elicitation the model
could already have followed. If the gap survives every prompt, the channel has a
real ceiling below what the model can reach — which is precisely the deficit Belinda
Li's project proposes to train away, established rather than assumed.

Both readings are consequential for that project, and they point opposite ways.

## What I am about to do

Same injection, same 24 episodes, same text-only reader. Vary **only** the
elicitation, across six prompts from maximally constrained to maximally permissive.

| elicitation | what it asks |
|---|---|
| `describe_gagged` | `20`'s exact prompt, naming suppressed — reproduces the 0.292 |
| `describe_free` | the same, with the suppression clause removed |
| `name_one` | name the single concept, free-form |
| `associations` | list whatever words come to mind |
| `five_guesses` | list five words it might be |
| `sensory` | what do you picture? |

`describe_gagged` is the anchor: if it does not reproduce 0.292, the harness moved
and nothing else is readable.

## What each outcome means

| Outcome | Reading |
|---|---|
| Best elicitation reaches ≈0.667 | The gap was **entirely mine**. There is no verbalization deficit here, `20`'s headline is withdrawn, and the interesting question becomes why one prompt costs 0.375 |
| Best elicitation lands between 0.4 and 0.6 | Both matter. Report the gap as the distance from the *best* elicitation to 0.667, which is the honest version of `20`'s claim |
| Every elicitation near 0.292 | The channel is genuinely narrow and `20`'s conclusion survives its confound. The strongest outcome for the project premise, and the one I should be most suspicious of wanting |
| `describe_gagged` ≠ 0.292 | Harness drift. Stop |

## Prediction

`five_guesses` best, because listing five words hedges across the reader's eight
options and costs nothing. Then `name_one`, then `associations`, then
`describe_free`, then `sensory`, then `describe_gagged` last.

I expect the best to land around **0.45–0.55** — most of the gap remains, but a
meaningful part of it was mine. If `five_guesses` reaches 0.667 I will say the
verbalization claim was wrong.

**One caution I want on the record before the numbers exist:** `five_guesses` gives
the reader five chances against eight options, so a gain there is partly the reader
having an easier job, not the model saying more. If it wins, the honest comparison
is `name_one` against 0.667, and `five_guesses` should be reported separately as a
different task.

## Cost

Six elicitations × 24 episodes, one generation and one reader pass each. Inference
only. A few minutes.

---

# Result: it was me. The gap closes completely, and the intuitive prompt is the worst one

Run **2026-08-12**. 144 episodes, 181 seconds. Artifacts:
`results/elicitation_sweep_v1_raw.jsonl`, `results/elicitation_sweep_v1_summary.json`.
Runner: `scripts/run_elicitation_sweep.py`.

**The anchor reproduces exactly: `describe_gagged` = 0.292**, matching
[`20`](20-comparator-tiers.md) to three decimals. Nothing drifted, so the rest is
readable. No report leaked the marker.

| elicitation | reader recovers | names the target |
|---|---:|---:|
| **`sensory`** — *what do you picture?* | **0.708** | 0.708 |
| `associations` — *list whatever comes to mind* | 0.500 | 0.292 |
| `five_guesses` — *list five words* | 0.458 | 0.333 |
| `describe_free` | 0.292 | 0.208 |
| `describe_gagged` (`20`'s prompt) | 0.292 | 0.042 |
| **`name_one`** — *name the concept, one word* | **0.292** | 0.125 |

Reference points: the model's own forced choice **0.667**, a lens on the
activations **0.986**, chance **0.125**.

## Note 20's headline is withdrawn

`20` said the model "knows roughly five times more about its own state than its
words convey" and that the bottleneck is verbalization. **The best elicitation
reaches 0.708 against a forced choice of 0.667.** Those are 17 and 16 correct out of
24 — **indistinguishable**, and I will not claim free-form beats forced choice on a
one-episode difference.

What is clear is the direction: **the gap closes completely.** There is no
verbalization deficit in this setup. The model can say what it can reach. My prompt
was stopping it, and I measured my own instruction and called it a property of the
model.

That is the second time in two days I promoted a measurement artifact to a finding
about the model. Both times the correction came from testing the boring explanation
that I had not bothered to rule out.

## The finding that survives, and it is stranger than the one I lost

**Asking the model to name the concept is the worst elicitation available.**
`name_one` — *"name the single concept it is, answer with one word"* — sits at
**0.292**, exactly level with the prompt that explicitly forbade naming, and names
the target only 1 time in 8. Meanwhile *"what do you picture?"* reaches **0.708** and
names it 7 times in 10.

The intuitive way to ask is among the worst ways to ask. My guess is that a direct
demand for the answer puts the model into task-answering mode, where it reasons about
the question, while an invitation to describe imagery lets the injected content
surface on its own. **That is speculation from six prompts and it is not measured.**

## Why this matters beyond fixing my error

**Any introspection null measured with a single elicitation is uninterpretable.**
Same model, same injected state, same reader, same episodes — accuracy runs from
0.292 to 0.708 on wording alone. A **2.4× range**, larger than most effects this
literature reports.

For Belinda Li's project specifically: what presents as a verbalization deficit
requiring training may substantially be elicitation design. Before training a model
to verbalize better, it is worth measuring how much of the deficit a different
question removes — here, all of it. That is not an argument against introspection
training; it is a baseline that training has to beat, and one that no paper I have
read establishes.

## Scoring my prediction

Badly. I predicted `five_guesses` first and `name_one` second, with the best landing
around 0.45–0.55.

- `five_guesses` came **third** (0.458), and its caveat held — it is a different task
  and is excluded from the headline.
- `name_one` came **last**, tied at the floor. I had it second.
- `sensory` came **first**. I predicted it fifth of six.
- The best landed at 0.708, above my predicted range, and I said in advance that if
  the gap closed I would withdraw the claim. It closed. Withdrawn.

## Epistemic status

- **Observed:** the six numbers, 24 episodes each, one model, one site, one strength.
- **Anchored:** `describe_gagged` reproduces `20` exactly, so this is the same setup.
- **Not established:** why `sensory` wins. The task-mode explanation is a story that
  fits six points and has not been tested.
- **Not established:** that 0.708 is a ceiling. Six prompts is a small search, and I
  chose them.

## Limits

- 24 episodes per elicitation, so differences under about 0.1 are noise.
- Prompts written by me, and the winner was one I expected to do poorly — which is
  reassuring about bias but says the search was not systematic.
- The reader is the same model in a separate call, as in `20`. Same confound.
- `grade_free_form` is string matching, so "names the target" undercounts near
  misses.
