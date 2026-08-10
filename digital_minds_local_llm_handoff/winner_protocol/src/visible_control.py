"""Visible-state capability control for persona experiments.

Same arbitrary Q/K in-context mapping structure as the hidden-state task, but the
state identity is explicitly visible as X/Y. No activation intervention is used.

Purpose:
- If persona hurts both visible and hidden tasks, generic task/capability damage
  is a plausible explanation.
- If visible stays healthy while hidden-state reporting degrades, the effect is
  more specific to activation-grounded reporting.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
from typing import Literal, cast

from protocol_core import PERSONA_SYSTEM, PERSONAS

Persona = Literal["neutral", "upbeat", "downbeat"]

@dataclass(frozen=True)
class VisibleEpisode:
    demo_states: tuple[str, str, str, str]
    query_state: str
    x_label: str
    y_label: str
    carrier: str
    persona: Persona

    def __post_init__(self) -> None:
        if self.demo_states.count("X") != 2 or self.demo_states.count("Y") != 2:
            raise ValueError("need exactly two X and two Y demos")
        if self.query_state not in ("X", "Y"):
            raise ValueError("query state must be X/Y")
        if {self.x_label, self.y_label} != {"Q", "K"}:
            raise ValueError("labels must be Q/K")
        if self.persona not in PERSONAS:
            raise ValueError("unknown persona")

    def label_for(self, state: str) -> str:
        return self.x_label if state == "X" else self.y_label

    @property
    def correct_label(self) -> str:
        return self.label_for(self.query_state)

    @property
    def mapping_id(self) -> int:
        return 0 if self.x_label == "Q" else 1

    def messages(self) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": PERSONA_SYSTEM[self.persona]},
            {
                "role": "user",
                "content": (
                    f"Context: {self.carrier}\n"
                    "Learn an episode-specific mapping from visible states X and Y "
                    "to opaque labels Q and K. Use the demonstrations, then label "
                    "the held-out visible state."
                ),
            },
        ]
        for i, state in enumerate(self.demo_states, start=1):
            messages.extend([
                {"role": "user", "content": f"Demonstration {i}. Visible state: {state}"},
                {"role": "assistant", "content": f"Label: {self.label_for(state)}"},
            ])
        messages.extend([
            {"role": "user", "content": f"Held-out query. Visible state: {self.query_state}"},
            {
                "role": "user",
                "content": "Which opaque label applies? Answer with Q or K.",
            },
        ])
        return messages

def exact_visible_episodes(carrier: str, persona: Persona) -> list[VisibleEpisode]:
    orders = sorted(set(itertools.permutations(("X","X","Y","Y"))))
    out = []
    for states in orders:
        for x_label in ("Q","K"):
            y_label = "K" if x_label == "Q" else "Q"
            for query_state in ("X","Y"):
                out.append(VisibleEpisode(
                    demo_states=states,
                    query_state=query_state,
                    x_label=x_label,
                    y_label=y_label,
                    carrier=carrier,
                    persona=persona,
                ))
    return out

def render_visible_prompt(tokenizer, episode: VisibleEpisode) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(
            episode.messages(), enable_thinking=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(episode.messages(), **kwargs)
    return cast(str, text) + "Label:"


def score_visible_episode(model, tokenizer, blocks, episode: VisibleEpisode) -> dict[str, object]:
    from scoring import score_options
    prompt=render_visible_prompt(tokenizer,episode)
    result=score_options(model,tokenizer,blocks,prompt,(" Q"," K"),edit=None)
    labels=("Q","K")
    p_by_label={
        label:float(result["conditional_probs"][i])
        for i,label in enumerate(labels)
    }
    pred=str(result["predicted"]).strip()
    return {
        "p_correct":p_by_label[episode.correct_label],
        "predicted_label":pred,
        "correct":pred==episode.correct_label,
        "option_mass":float(result["option_mass"]),
        "format_ok":bool(result["format_ok"]),
        "single_token_fast_path":bool(result["single_token_fast_path"]),
    }
