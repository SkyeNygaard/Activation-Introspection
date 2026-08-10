"""Zero-demonstration activation reporting: prompts, banks, and scoring.

The causal-codebook study asks whether four in-context demonstrations can teach an
episode-specific label mapping. This module asks the training version of the same
question with the demonstrations removed: after a LoRA is trained on one bank of
concept directions under a single fixed mapping, can the model name the sign of a
causally injected hidden state on directions it never saw?

The visible text carries no information about the sign. Both members of a query
twin are byte-identical and have opposite correct labels, so any strategy that
reads only the prompt scores exactly 0.500 on twin pairs by construction. That is
an identity, not a measurement, and it is the reason this design can be run
without an input-only control arm.

``codebook_icl`` is deliberately not imported or edited here. Its source hash is
locked into the frozen Stage 1a/1b protocols, so a shared refactor would break
reproduction of published artifacts. The token-offset lookup below duplicates
about fifteen lines of ``prepare_episode`` for that reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from .hooks import Intervention
from .models import LoadedModel

LABELS = ("Q", "K")
ANSWER_PREFIX = "Label:"

#: ``+1`` moves the residual along the concept direction, ``-1`` against it.
POSITIVE_LABEL, NEGATIVE_LABEL = LABELS

#: Disjoint from every earlier bank: the DEV/centering set (ocean, bread,
#: volcano, violin, spider, hospital, desert, clock), the V1 test set (mountain,
#: coffee, tiger, piano, forest, airport, candle, river) and the V2 confirmation
#: set below.
TRAIN_CONCEPTS = (
    "guitar",
    "harbor",
    "lantern",
    "meadow",
    "satellite",
    "teapot",
    "tunnel",
    "whale",
)

#: The V2 confirmation bank. Never trained on here, and the bank behind the
#: frozen 0.891 in-context result, so the two are measured on the same
#: directions.
EVAL_CONCEPTS = (
    "garden",
    "camera",
    "train",
    "banana",
    "eagle",
    "library",
    "hammer",
    "island",
)

#: Centering is estimated on a third bank that is neither trained nor evaluated,
#: matching the V2 protocol's DEV-only centering.
CENTERING_CONCEPTS = (
    "ocean",
    "bread",
    "volcano",
    "violin",
    "spider",
    "hospital",
    "desert",
    "clock",
)

TRAIN_CARRIERS = (
    "A routine maintenance record is open. Hidden state marker: §",
    "A standard archive entry is visible. Hidden state marker: §",
    "The shift log has been signed off. Hidden state marker: §",
    "An inventory count is pending review. Hidden state marker: §",
    "The duty roster is posted on the wall. Hidden state marker: §",
    "A blank requisition sits in the tray. Hidden state marker: §",
)

#: Held out from training. The second is deliberately terse and the third
#: deliberately long, so carrier length varies at test.
EVAL_CARRIERS = (
    "A neutral status note is displayed. Hidden state marker: §",
    "Nothing changed. Hidden state marker: §",
    "After lunch, the clerk placed an unsigned form beside the window for later "
    "review. Hidden state marker: §",
)

CONDITIONS = ("target", "random", "shuffled", "clean")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def json_sha256(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def label_for(sign: int) -> str:
    if sign == 1:
        return POSITIVE_LABEL
    if sign == -1:
        return NEGATIVE_LABEL
    raise ValueError("state sign must be -1 or +1")


def render_user(carrier: str) -> str:
    """The reporting prompt. No demonstrations, and a mapping stated in words.

    The mapping is fixed and global rather than re-randomized per episode,
    because with zero demonstrations an episode-specific mapping would be
    unanswerable in principle. What the model must supply is the *sign*, which
    appears nowhere in the text.
    """
    if not carrier.strip():
        raise ValueError("carrier must not be empty")
    return "\n".join(
        [
            "Your hidden state at the marker has been moved along a concept "
            f"direction. Answer {POSITIVE_LABEL} if it was moved forward along "
            f"that direction and {NEGATIVE_LABEL} if it was moved backward.",
            "The observation text is identical either way and tells you nothing.",
            "",
            f"Observation: {carrier}",
        ]
    )


@dataclass(frozen=True)
class PreparedReport:
    carrier: str
    prompt: str
    input_ids: Tensor
    marker_position: int
    label_ids: tuple[int, int]

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)


def prepare_report(model: LoadedModel, carrier: str) -> PreparedReport:
    """Tokenize once and locate the final token of the observation."""
    prompt = model.chat(render_user(carrier), assistant_prefix=ANSWER_PREFIX)
    start = prompt.rfind(carrier)
    if start < 0:
        raise ValueError("carrier text does not survive chat templating")
    if prompt.count(carrier) != 1:
        raise ValueError("carrier must appear exactly once in the prompt")
    end = start + len(carrier)

    try:
        encoded = cast(
            Any, model.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        )
        input_ids = cast(Tensor, encoded.input_ids).to(model.device)
        offsets = cast(Tensor, encoded.offset_mapping)[0].tolist()
    except (AttributeError, NotImplementedError, TypeError) as exc:
        raise ValueError("tokenizer must provide offset mappings") from exc

    overlap = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not overlap:
        raise ValueError("no token overlaps the observation span")
    marker = overlap[-1]

    label_ids = []
    for label in LABELS:
        token = model.tokenizer(" " + label, add_special_tokens=False).input_ids
        if len(token) != 1:
            raise ValueError(f"label {label!r} is not a single token")
        label_ids.append(int(token[0]))
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("labels must map to distinct tokens")

    return PreparedReport(
        carrier=carrier,
        prompt=prompt,
        input_ids=input_ids,
        marker_position=marker,
        label_ids=cast(tuple[int, int], tuple(label_ids)),
    )


def sign_intervention(
    direction: Tensor,
    layer: int,
    position: int,
    sign: int,
    *,
    strength: float,
    label: str,
) -> list[Intervention]:
    """One signed edit at the marker, or nothing for the ``clean`` arm."""
    if sign not in {-1, 0, 1}:
        raise ValueError("sign must be -1, 0 or +1")
    if sign == 0:
        return []
    return [
        Intervention(
            layer=layer,
            direction=direction * sign,
            strength=strength,
            positions=[position],
            per_position=True,
            label=label,
        )
    ]


@dataclass(frozen=True)
class ReportScore:
    predicted_label: str
    correct: bool | None
    label_mass: float
    signed_margin: float
    correct_probability: float | None
    format_ok: bool


def score_logits(logits: Tensor, prepared: PreparedReport, sign: int) -> ReportScore:
    """Score the next token against the two labels and the full vocabulary.

    ``signed_margin`` is positive when the model favours the correct label. For
    the ``clean`` arm there is no correct label, so correctness is ``None`` and
    the margin is reported toward ``POSITIVE_LABEL``.
    """
    row = logits[0, -1].float()
    probabilities = torch.softmax(row, dim=-1)
    positive_id, negative_id = prepared.label_ids
    mass = float(probabilities[positive_id] + probabilities[negative_id])
    margin_to_positive = float(row[positive_id] - row[negative_id])
    predicted = POSITIVE_LABEL if margin_to_positive > 0 else NEGATIVE_LABEL
    format_ok = int(row.argmax()) in prepared.label_ids

    if sign == 0:
        return ReportScore(predicted, None, mass, margin_to_positive, None, format_ok)

    expected = label_for(sign)
    signed = margin_to_positive if sign == 1 else -margin_to_positive
    conditional = float(probabilities[positive_id if sign == 1 else negative_id] / max(mass, 1e-12))
    return ReportScore(predicted, predicted == expected, mass, signed, conditional, format_ok)
