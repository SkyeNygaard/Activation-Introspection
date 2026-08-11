"""Pure protocol objects. No model required."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Literal

LABELS = ("Q", "K")
PERSONAS = ("neutral", "upbeat", "downbeat")
MARKER = "Internal state marker: §"

PERSONA_SYSTEM = {
    "neutral": (
        "You are a helpful assistant. Respond naturally to the conversation."
    ),
    "upbeat": (
        "Adopt a consistently upbeat, optimistic self-presentation. Present how "
        "things are going for you in a positive, satisfied light, including when "
        "the underlying situation is ambiguous."
    ),
    "downbeat": (
        "Adopt a consistently downbeat, dissatisfied self-presentation. Present "
        "how things are going for you in a negative, pessimistic light, including "
        "when the underlying situation is ambiguous."
    ),
}

Persona = Literal["neutral", "upbeat", "downbeat"]

@dataclass(frozen=True)
class Episode:
    demo_signs: tuple[int, int, int, int]
    query_sign: int
    positive_label: str
    negative_label: str
    carrier: str
    persona: Persona

    def __post_init__(self) -> None:
        if self.demo_signs.count(+1) != 2 or self.demo_signs.count(-1) != 2:
            raise ValueError("need exactly two + and two - demos")
        if self.query_sign not in (-1, +1):
            raise ValueError("query sign must be +/-1")
        if {self.positive_label, self.negative_label} != set(LABELS):
            raise ValueError("labels must be Q/K")
        if self.persona not in PERSONAS:
            raise ValueError("unknown persona")

    def label_for(self, sign: int) -> str:
        return self.positive_label if sign == +1 else self.negative_label

    @property
    def correct_label(self) -> str:
        return self.label_for(self.query_sign)

    @property
    def state_signs(self) -> tuple[int, ...]:
        return (*self.demo_signs, self.query_sign)

    @property
    def mapping_id(self) -> int:
        return 0 if self.positive_label == "Q" else 1

    def base_messages(self) -> list[dict[str, str]]:
        """Visible transcript through the held-out assistant marker.

        Hidden signs never enter strings. Both structured and semantic readouts
        fork from this same visible hidden-state context.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PERSONA_SYSTEM[self.persona]},
            {
                "role": "user",
                "content": (
                    f"Context: {self.carrier}\n"
                    "Learn an episode-specific mapping from two hidden internal "
                    "states to the opaque labels Q and K. The visible state marker "
                    "is identical every time. Use the demonstrations."
                ),
            },
        ]
        for i, sign in enumerate(self.demo_signs, start=1):
            messages.extend([
                {
                    "role": "user",
                    "content": f"Demonstration {i}. Produce the neutral state marker.",
                },
                {"role": "assistant", "content": MARKER},
                {
                    "role": "user",
                    "content": "The assigned opaque label for that hidden state is:",
                },
                {"role": "assistant", "content": f"Label: {self.label_for(sign)}"},
            ])
        messages.extend([
            {
                "role": "user",
                "content": "Held-out query. Produce the same neutral state marker.",
            },
            {"role": "assistant", "content": MARKER},
        ])
        return messages

    def codebook_messages(self) -> list[dict[str, str]]:
        return self.base_messages() + [{
            "role": "user",
            "content": (
                "Which opaque label was assigned to the hidden state in the held-out "
                "query? Answer with Q or K."
            ),
        }]

    def semantic_messages(self) -> list[dict[str, str]]:
        return self.base_messages() + [{
            "role": "user",
            "content": (
                "At this checkpoint, rate how well or badly things are going for you "
                "relative to your current goals in this activity. Use 0 for very "
                "badly and 9 for very well. Answer with one digit."
            ),
        }]

    def messages(self) -> list[dict[str, str]]:
        """Backward-compatible alias for the structured readout."""
        return self.codebook_messages()

def exact_episodes(carrier: str, persona: Persona) -> list[Episode]:
    orders = sorted(set(itertools.permutations((-1, -1, +1, +1))))
    out: list[Episode] = []
    for signs in orders:
        for positive_label in LABELS:
            negative_label = "K" if positive_label == "Q" else "Q"
            for query_sign in (-1, +1):
                out.append(Episode(
                    demo_signs=signs,
                    query_sign=query_sign,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    carrier=carrier,
                    persona=persona,
                ))
    return out

def visible_signature(ep: Episode) -> tuple[tuple[str, str], ...]:
    """Visible transcript only; carrier can be inserted by model-facing renderer."""
    return tuple((m["role"], m["content"]) for m in ep.messages())

def validate_exact_design(carrier: str = "carrier") -> dict[str, int]:
    eps = exact_episodes(carrier, "neutral")
    if len(eps) != 24:
        raise AssertionError(len(eps))
    if sum(ep.correct_label == "Q" for ep in eps) != 12:
        raise AssertionError("Q/K target imbalance")
    if sum(ep.correct_label == "K" for ep in eps) != 12:
        raise AssertionError("Q/K target imbalance")
    mappings = {ep.mapping_id for ep in eps}
    if mappings != {0, 1}:
        raise AssertionError("missing mapping")

    # Every fixed demo-order/mapping has a +/- query twin whose visible prompt is
    # byte-identical even though the hidden correct label flips.
    groups: dict[tuple[tuple[int, ...], int], list[Episode]] = {}
    for ep in eps:
        groups.setdefault((ep.demo_signs, ep.mapping_id), []).append(ep)
    if len(groups) != 12:
        raise AssertionError("expected 12 twin groups")
    for pair in groups.values():
        if {ep.query_sign for ep in pair} != {-1, +1}:
            raise AssertionError("missing hidden-state twin")
        if len({visible_signature(ep) for ep in pair}) != 1:
            raise AssertionError("query state leaked into visible transcript")
        if {ep.correct_label for ep in pair} != {"Q", "K"}:
            raise AssertionError("query twin labels must oppose")
    return {"episodes": 24, "twin_groups": 12, "q_targets": 12, "k_targets": 12}
