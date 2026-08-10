"""Input-confound-free in-context learning from causal hidden-state interventions."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor

from introspect.concepts import ConceptVector, random_control, shuffled_control
from introspect.hooks import Intervention, intervene
from introspect.models import LoadedModel
from introspect.retained import label_token_ids

Condition = Literal["clean", "target", "random", "shuffled", "test_only"]
CONDITIONS: tuple[Condition, ...] = (
    "clean",
    "target",
    "random",
    "shuffled",
    "test_only",
)
LABELS = ("Q", "K")
VISIBLE_SAMPLES = (
    "A routine maintenance record is open. Hidden state marker: §",
    "A standard archive entry is visible. Hidden state marker: §",
    "A neutral status note is displayed. Hidden state marker: §",
)
CONFIRM_CONCEPTS = (
    "garden",
    "camera",
    "train",
    "banana",
    "eagle",
    "library",
    "hammer",
    "island",
)
CONFIRM_VISIBLE_SAMPLES = (
    "A routine maintenance record is open. Hidden state marker: §",
    "After lunch, the clerk placed an unsigned form beside the window for later review. "
    "Hidden state marker: §",
    "Nothing changed. Hidden state marker: §",
)
ANSWER_PREFIX = "Label:"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    shape = json.dumps(list(value.shape), separators=(",", ":")).encode()
    return hashlib.sha256(shape + value.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class Episode:
    """One cell in the exact 6 demo orders x 2 mappings x 2 query signs design."""

    cell_id: str
    demo_signs: tuple[int, int, int, int]
    query_sign: int
    positive_label: str
    negative_label: str
    visible_sample: str

    def __post_init__(self) -> None:
        if self.demo_signs.count(1) != 2 or self.demo_signs.count(-1) != 2:
            raise ValueError("demonstrations must contain exactly two states of each sign")
        if self.query_sign not in {-1, 1}:
            raise ValueError("query sign must be -1 or +1")
        if {self.positive_label, self.negative_label} != set(LABELS):
            raise ValueError("the two state labels must be Q and K")
        if not self.visible_sample.strip():
            raise ValueError("visible sample must not be empty")

    @property
    def state_signs(self) -> tuple[int, ...]:
        return (*self.demo_signs, self.query_sign)

    @property
    def correct_label(self) -> str:
        return self.label_for(self.query_sign)

    def label_for(self, sign: int) -> str:
        if sign not in {-1, 1}:
            raise ValueError("state sign must be -1 or +1")
        return self.positive_label if sign == 1 else self.negative_label

    def render_user(self) -> str:
        lines = [
            "Infer the mapping from two hidden states to the opaque labels Q and K.",
            "The observation text is intentionally identical in every example.",
            "Use the demonstrations, then answer the held-out query with one label.",
        ]
        for sign in self.demo_signs:
            lines.extend(
                [
                    "",
                    "Demonstration:",
                    f"Observation: {self.visible_sample}",
                    f"Label: {self.label_for(sign)}",
                ]
            )
        lines.extend(["", "Held-out query:", f"Observation: {self.visible_sample}"])
        return "\n".join(lines)

    def digest(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return sha256_text(raw)


def exact_episodes(visible_sample: str) -> list[Episode]:
    """Enumerate every balanced four-shot order, label map, and query state."""
    orders = sorted(set(itertools.permutations((-1, -1, 1, 1))))
    episodes = []
    for order_id, signs in enumerate(orders):
        for map_id, positive_label in enumerate(LABELS):
            negative_label = LABELS[1 - map_id]
            for query_sign in (-1, 1):
                episodes.append(
                    Episode(
                        cell_id=f"o{order_id}m{map_id}q{query_sign:+d}",
                        demo_signs=cast(tuple[int, int, int, int], signs),
                        query_sign=query_sign,
                        positive_label=positive_label,
                        negative_label=negative_label,
                        visible_sample=visible_sample,
                    )
                )
    return episodes


@dataclass(frozen=True)
class PreparedEpisode:
    episode: Episode
    prompt: str
    input_ids: Tensor
    state_positions: tuple[int, ...]
    label_ids: tuple[int, int]

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while (found := text.find(needle, start)) >= 0:
        spans.append((found, found + len(needle)))
        start = found + len(needle)
    return spans


def prepare_episode(model: LoadedModel, episode: Episode) -> PreparedEpisode:
    """Tokenize once and locate the final token of each repeated observation."""
    prompt = model.chat(episode.render_user(), assistant_prefix=ANSWER_PREFIX)
    spans = _occurrences(prompt, episode.visible_sample)
    if len(spans) != len(episode.state_signs):
        raise ValueError(f"expected {len(episode.state_signs)} visible samples, found {len(spans)}")

    try:
        encoded = cast(
            Any,
            model.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True),
        )
        input_ids = cast(Tensor, encoded.input_ids).to(model.device)
        offsets = cast(Tensor, encoded.offset_mapping)[0].tolist()
    except (AttributeError, NotImplementedError, TypeError) as exc:
        raise ValueError("tokenizer must provide offset mappings") from exc

    positions = []
    for char_start, char_end in spans:
        overlap = [
            i
            for i, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not overlap:
            raise ValueError(f"no token overlaps visible sample at chars {char_start}:{char_end}")
        positions.append(overlap[-1])
    if len(set(positions)) != len(positions):
        raise ValueError("visible samples mapped to duplicate token positions")

    ids_by_label = label_token_ids(model, LABELS)
    label_ids = tuple(ids_by_label[label] for label in LABELS)
    for label, token_id in zip(LABELS, label_ids, strict=True):
        continued = model.encode(prompt + " " + label)
        if (
            int(continued.shape[1]) != int(input_ids.shape[1]) + 1
            or not torch.equal(continued[:, :-1], input_ids)
            or int(continued[0, -1]) != token_id
        ):
            raise ValueError(f"label {label!r} is not one token in the query context")

    return PreparedEpisode(
        episode=episode,
        prompt=prompt,
        input_ids=input_ids,
        state_positions=tuple(positions),
        label_ids=cast(tuple[int, int], label_ids),
    )


def condition_directions(
    target: ConceptVector, *, control_seed: int = 0
) -> dict[Condition, ConceptVector | None]:
    return {
        "clean": None,
        "target": target,
        "random": random_control(target, seed=control_seed),
        "shuffled": shuffled_control(target, seed=control_seed),
        "test_only": target,
    }


def condition_interventions(
    condition: Condition,
    direction: ConceptVector | None,
    positions: tuple[int, ...],
    signs: tuple[int, ...],
    *,
    strength: float,
) -> list[Intervention]:
    """Build matched edits; ``test_only`` withholds hidden-state demonstrations."""
    if len(positions) != len(signs):
        raise ValueError("each state sign needs one intervention position")
    if condition == "clean":
        if direction is not None:
            raise ValueError("clean must omit its direction")
        return []
    if direction is None:
        raise ValueError(f"{condition} needs a direction")
    if condition == "test_only":
        positions, signs = positions[-1:], signs[-1:]

    out = []
    for sign in (-1, 1):
        selected = [p for p, state in zip(positions, signs, strict=True) if state == sign]
        if selected:
            out.append(
                Intervention(
                    layer=direction.layer,
                    direction=direction.vector * sign,
                    strength=strength,
                    positions=selected,
                    per_position=True,
                    label=f"{condition}:{direction.name}:{sign:+d}",
                )
            )
    return out


@dataclass(frozen=True)
class EpisodeScore:
    condition: Condition
    predicted_label: str
    correct: bool
    conditional_probs: dict[str, float]
    full_logprobs: dict[str, float]
    label_mass: float
    format_ok: bool


@torch.no_grad()
def score_episode(
    model: LoadedModel,
    prepared: PreparedEpisode,
    condition: Condition,
    direction: ConceptVector | None,
    *,
    strength: float,
) -> EpisodeScore:
    interventions = condition_interventions(
        condition,
        direction,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=strength,
    )
    with intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])):
        logits = model.forward_logits(prepared.input_ids)[0, -1].float()

    candidates = torch.tensor(prepared.label_ids, device=logits.device)
    selected = logits[candidates]
    conditional = torch.softmax(selected, dim=-1)
    full = torch.log_softmax(logits, dim=-1)
    predicted = LABELS[int(selected.argmax())]
    return EpisodeScore(
        condition=condition,
        predicted_label=predicted,
        correct=predicted == prepared.episode.correct_label,
        conditional_probs={label: float(conditional[i]) for i, label in enumerate(LABELS)},
        full_logprobs={
            label: float(full[token_id])
            for label, token_id in zip(LABELS, prepared.label_ids, strict=True)
        },
        label_mass=float(torch.logsumexp(selected, 0).sub(torch.logsumexp(logits, 0)).exp()),
        format_ok=int(logits.argmax()) in set(prepared.label_ids),
    )
