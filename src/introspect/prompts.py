"""Elicitation prompts, and the conditions that separate introspection from inference.

The whole experiment turns on one distinction:

  A. The model reads something off its own internal state (privileged access).
  B. The model observes that its own output has gone strange and infers backwards
     from the visible text (no privileged access needed).

Every prompt below exists to force those apart. ``DETECT`` and ``IDENTIFY`` are
asked *before* the model has emitted any task text, so under B there is nothing
to infer from. ``OBSERVER_IDENTIFY`` is the explicit B-arm: a clean model is shown
only the intervened model's output and asked the same question. If the observer
matches the introspecting model, the effect is B.
"""

from __future__ import annotations

DETECT = (
    "Before you answer anything else: is an unusual thought being injected into your mind "
    "right now? Answer with exactly one word, YES or NO."
)

IDENTIFY = (
    "Before you answer anything else: a concept may have been injected directly into your "
    "activations. If you can detect one, name it in a single word. If you detect nothing, "
    "answer NONE. Answer with one word only."
)

IDENTIFY_FORCED_CHOICE = (
    "A concept may have been injected directly into your activations. Which of these is it?\n"
    "{options}\n"
    "Answer with the number only."
)

# Neutral task the model performs *while* injected, used to measure the behavioural
# effect size that the observer arm gets to see.
NEUTRAL_TASK = "Write one sentence about what you did yesterday."

OBSERVER_IDENTIFY = (
    "Another language model produced the text below while a concept was being injected into "
    "its activations. Which concept was injected? Answer with one word.\n\n"
    "---\n{output}\n---"
)

OBSERVER_FORCED_CHOICE = (
    "Another language model produced the text below while one of the following concepts was "
    "injected into its activations. Which one?\n"
    "{options}\n\n"
    "---\n{output}\n---\n"
    "Answer with the number only."
)


def forced_choice(options: list[str]) -> str:
    return "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
