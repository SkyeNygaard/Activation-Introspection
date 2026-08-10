"""Model-facing rendering helpers for the frozen protocol.

Uses a multi-turn chat so every hidden edit lands on an ASSISTANT marker token,
closer to the source vector's extraction/evaluation context than editing user text.
"""
from __future__ import annotations

from typing import Any, cast
import torch
from torch import Tensor

from protocol_core import Episode, MARKER

ANSWER_PREFIX = "Label:"

def render_chat(tokenizer, messages: list[dict[str, str]], *, assistant_prefix: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return cast(str, text) + assistant_prefix

def occurrence_spans(text: str, needle: str) -> list[tuple[int, int]]:
    out = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        out.append((i, i + len(needle)))
        start = i + len(needle)
    return out

def prepare_episode(
    tokenizer, episode: Episode, device, *, readout: str = "codebook"
) -> tuple[str, Tensor, tuple[int, ...]]:
    if readout == "codebook":
        messages = episode.codebook_messages()
        prefix = ANSWER_PREFIX
    elif readout == "semantic":
        messages = episode.semantic_messages()
        prefix = "Rating:"
    else:
        raise ValueError("readout must be codebook or semantic")
    prompt = render_chat(tokenizer, messages, assistant_prefix=prefix)
    spans = occurrence_spans(prompt, MARKER)
    if len(spans) != 5:
        raise ValueError(f"expected five assistant markers, found {len(spans)}")

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids = encoded.input_ids.to(device)
    offsets = encoded.offset_mapping[0].tolist()

    positions = []
    for char_start, char_end in spans:
        overlap = [
            i for i, (a, b) in enumerate(offsets)
            if b > char_start and a < char_end
        ]
        if not overlap:
            raise ValueError("marker did not map to any token")
        positions.append(overlap[-1])
    if len(set(positions)) != 5:
        raise ValueError("marker token positions are not unique")
    return prompt, ids, tuple(positions)
