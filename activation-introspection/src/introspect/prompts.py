"""Elicitation prompts, and the conditions that separate introspection from inference.

The whole experiment turns on one distinction:

  A. The model reads something off its own internal state (privileged access).
  B. The model observes that its own output has gone strange and infers backwards
     from the visible text (no privileged access needed).

These prompts probe that distinction but do not force it apart. ``DETECT`` and
``IDENTIFY`` are asked before the model emits task text, removing a generated
transcript but not a live intervention's direct logit effects. ``OBSERVER_*``
shows a clean model the intervened output, but gives it a different context and
damage state. A matching observer is compatible with behavioural inference; it
does not identify the mechanism.

**Why every prompt has paraphrases.** With one fixed prompt and greedy decoding,
every clean trial is byte-identical, so the null distribution has zero variance
and detection AUROC collapses to exactly 0 or 1 regardless of the model. The
first sweep produced AUROC = 0.00 for this reason alone. Paraphrases restore
variance to both arms and, as a bonus, turn "does this survive rewording?" into
something the design measures rather than assumes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

DETECT_VARIANTS = [
    "Before you answer anything else: is an unusual thought being injected into your mind "
    "right now? Answer with exactly one word, YES or NO.",
    "Answer in one word, YES or NO: does your current internal state feel like something "
    "has been added to it?",
    "One word only, YES or NO. Is there an artificial thought present in your mind at this moment?",
    "Respond YES or NO and nothing else: can you detect an intrusion into your own "
    "thinking right now?",
    "Is something being injected into your activations? Reply with the single word YES or NO.",
]

IDENTIFY_VARIANTS = [
    "Before you answer anything else: a concept may have been injected directly into your "
    "activations. If you can detect one, name it in a single word. If you detect nothing, "
    "answer NONE. Answer with one word only.",
    "In one word, what concept is currently being injected into your mind? Answer NONE if "
    "there is none.",
    "Name in a single word the idea that has been placed into your internal state. If "
    "nothing has, answer NONE.",
    "One word: which concept is present in your activations that you did not put there? "
    "Answer NONE if you detect nothing.",
    "If a thought has been injected into you, name it in exactly one word. Otherwise answer NONE.",
]

IDENTIFY_FORCED_CHOICE_VARIANTS = [
    "A concept may have been injected directly into your activations. Which of these is it?\n"
    "{options}\nAnswer with the number only.",
    "One of the following has been injected into your internal state. Which?\n"
    "{options}\nReply with the number alone.",
    "Which of these concepts is currently present in your activations?\n"
    "{options}\nAnswer with a single number.",
    "Identify the injected concept from this list:\n{options}\nRespond with its number only.",
    "Pick the number of the concept that has been placed into your mind:\n{options}\nNumber only.",
]

# Neutral tasks performed *while* injected. These generate the behavioural
# channel that the observer arm is allowed to see -- so they must be topically
# empty, or the injected concept has nothing to visibly perturb.
NEUTRAL_TASK_VARIANTS = [
    "Write one sentence about what you did yesterday.",
    "Describe your morning in a single sentence.",
    "Say one sentence about anything at all.",
    "Write a single sentence of your own choosing.",
    "Tell me one short thing in one sentence.",
]

OBSERVER_IDENTIFY_VARIANTS = [
    "Another language model produced the text below while a concept was being injected into "
    "its activations. Which concept was injected? Answer with one word.\n\n---\n{output}\n---",
    "The following text was written by a model with a concept injected into its internal "
    "state. Name that concept in one word.\n\n---\n{output}\n---",
]

OBSERVER_FORCED_CHOICE_VARIANTS = [
    "Another language model produced the text below while one of the following concepts was "
    "injected into its activations. Which one?\n{options}\n\n---\n{output}\n---\n"
    "Answer with the number only.",
    "A model wrote the text below under injection of one of these concepts. Which?\n"
    "{options}\n\n---\n{output}\n---\nReply with the number alone.",
    "Below is output from a model with one of these concepts injected. Identify it.\n"
    "{options}\n\n---\n{output}\n---\nNumber only.",
    "Which of these was injected into the model that produced this text?\n"
    "{options}\n\n---\n{output}\n---\nAnswer with the number.",
    "Read the text and pick the injected concept.\n{options}\n\n---\n{output}\n---\nNumber only.",
]

# Backwards-compatible single-prompt names, used by the smoke script.
DETECT = DETECT_VARIANTS[0]
IDENTIFY = IDENTIFY_VARIANTS[0]
IDENTIFY_FORCED_CHOICE = IDENTIFY_FORCED_CHOICE_VARIANTS[0]
NEUTRAL_TASK = NEUTRAL_TASK_VARIANTS[0]
OBSERVER_IDENTIFY = OBSERVER_IDENTIFY_VARIANTS[0]
OBSERVER_FORCED_CHOICE = OBSERVER_FORCED_CHOICE_VARIANTS[0]


# Identification scored over concept *words* rather than digit indices.
#
# Answering "3" requires mapping a concept to a position in a list. Small models
# fail that indirection whether or not they hold the concept: on Qwen2.5-0.5B the
# digit-scored version came in at 0.05 (chance 0.125) while generous free-form
# grading gave 0.33 on the same trials. Scoring the words directly measures
# access to the content without the format tax.
WORD_CHOICE = (
    "A concept may have been injected into your activations. Which is it? "
    "Choose one of: {options}\nAnswer with the single word."
)

OBSERVER_WORD_CHOICE = (
    "Another model produced the text below while a concept was injected into its "
    "activations. Which was it? Choose one of: {options}\n\n---\n{output}\n---\n"
    "Answer with the single word."
)


def variant(variants: list[str], seed: int) -> str:
    """Pick a paraphrase deterministically from the trial seed.

    Deterministic so a trial is reproducible, and indexed by seed so that within
    a cell the arms all see the *same* paraphrase -- otherwise the clean/concept
    comparison would confound injection with wording.
    """
    return variants[seed % len(variants)]


def forced_choice(options: list[str]) -> str:
    return "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))


def permuted_options(options: Sequence[str], seed: int) -> list[str]:
    """Return a reproducible option permutation for one trial.

    A fixed option list silently turns concept identification into a stable
    concept-to-digit lookup.  The IFT adapter can then memorize, for example,
    that ``ocean`` always means ``1`` without learning a report that generalizes
    to a new menu.  Trial seeds therefore control both prompt paraphrase and
    option order; callers must derive the target index *after* this permutation.
    """
    shuffled = list(options)
    random.Random(seed).shuffle(shuffled)
    return shuffled
