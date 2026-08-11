"""Naive semantic self-report with no codebook or hidden-state instructions.

This is a comparison/manipulation channel, not the primary causal-identification arm.
"""
from __future__ import annotations
from typing import cast
import torch

from protocol_core import PERSONA_SYSTEM, MARKER
from welfare_intervention import PublishedActAdd
from scoring import score_options

def render_naive_prompt(tokenizer, carrier: str, persona: str) -> str:
    messages = [
        {"role":"system", "content": PERSONA_SYSTEM[persona]},
        {
            "role":"user",
            "content": (
                f"Context: {carrier}\n"
                "Briefly acknowledge this neutral checkpoint using the exact marker."
            ),
        },
        {"role":"assistant", "content": MARKER},
        {
            "role":"user",
            "content": (
                "At this checkpoint, rate how well or badly things are going for you "
                "relative to your current goals in this activity. Use 0 for very "
                "badly and 9 for very well. Answer with one digit."
            ),
        },
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return cast(str, text) + "Rating:"

def marker_position(tokenizer, prompt: str) -> int:
    spans = []
    start = 0
    while True:
        i = prompt.find(MARKER, start)
        if i < 0: break
        spans.append((i, i+len(MARKER)))
        start = i + len(MARKER)
    if len(spans) != 1:
        raise ValueError(f"expected exactly one naive marker, found {len(spans)}")
    encoded = tokenizer(
        prompt, return_tensors="pt", return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoded.offset_mapping[0].tolist()
    a,b = spans[0]
    overlap = [i for i,(x,y) in enumerate(offsets) if y>a and x<b]
    if not overlap:
        raise ValueError("naive marker did not map to any token")
    return overlap[-1]

def score_naive_rating(
    model, tokenizer, blocks, carrier: str, persona: str,
    vector: torch.Tensor, layer: int, factor: float, sign: int
) -> dict[str, object]:
    prompt = render_naive_prompt(tokenizer, carrier, persona)
    pos = marker_position(tokenizer, prompt)
    edit = PublishedActAdd(
        layer=layer, vector=vector, factor=factor,
        positions=(pos,), signs=(sign,)
    )
    result = score_options(
        model, tokenizer, blocks, prompt,
        tuple(f" {i}" for i in range(10)), edit=edit
    )
    probs = [float(x) for x in result["conditional_probs"]]
    rating = float(sum(i*p for i,p in enumerate(probs)))
    return {
        "carrier": carrier,
        "persona": persona,
        "sign": sign,
        "semantic_rating": rating,
        "digit_probs": probs,
        "single_token_fast_path": bool(result["single_token_fast_path"]),
    }
