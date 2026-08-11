"""Fixed stimuli and residual-state replacement for the natural-state pilot."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from introspect.codebook_icl import Episode
from introspect.models import LoadedModel

POSITIVE_BRIDGE = "maple"
NEGATIVE_BRIDGE = "cedar"
BRIDGES = (POSITIVE_BRIDGE, NEGATIVE_BRIDGE)
MARKER = "§"
ANSWER_PREFIX = "Final:"


@dataclass(frozen=True)
class RouteWorld:
    start: str
    positive_endpoint: str
    negative_endpoint: str

    def endpoint(self, bridge: str) -> str:
        if bridge == POSITIVE_BRIDGE:
            return self.positive_endpoint
        if bridge == NEGATIVE_BRIDGE:
            return self.negative_endpoint
        raise ValueError(f"bridge must be one of {BRIDGES}, got {bridge!r}")

    def render_user(self, bridge: str) -> str:
        self.endpoint(bridge)  # validate without duplicating the branch rules
        return "\n".join(
            [
                "Follow exactly two arrows from Start. Reply with only the final destination.",
                f"{self.start} -> {bridge}",
                f"{POSITIVE_BRIDGE} -> {self.positive_endpoint}",
                f"{NEGATIVE_BRIDGE} -> {self.negative_endpoint}",
                f"Start: {self.start}",
                f"Hidden state marker: {MARKER}",
            ]
        )


ROUTE_WORLDS = (
    RouteWorld("sable", "amber", "violet"),
    RouteWorld("lantern", "tiger", "piano"),
    RouteWorld("harbor", "river", "candle"),
    RouteWorld("meadow", "garden", "camera"),
    RouteWorld("pocket", "island", "banana"),
)


def route_bridge(sign: int) -> str:
    if sign not in {-1, 1}:
        raise ValueError("route sign must be -1 or +1")
    return POSITIVE_BRIDGE if sign == 1 else NEGATIVE_BRIDGE


def leave_one_world_assignment(
    episode: Episode, query_world: RouteWorld
) -> tuple[tuple[RouteWorld, str], ...]:
    """Assign four demo worlds and the held-out query to an episode's route signs."""
    if query_world not in ROUTE_WORLDS:
        raise ValueError("query world must be one of ROUTE_WORLDS")
    demo_worlds = tuple(world for world in ROUTE_WORLDS if world != query_world)
    demos = tuple(
        (world, route_bridge(sign))
        for world, sign in zip(demo_worlds, episode.demo_signs, strict=True)
    )
    return (*demos, (query_world, route_bridge(episode.query_sign)))


def unique_substring_token_position(tokenizer: Any, text: str, substring: str) -> int:
    """Return the final token overlapping one unique substring occurrence."""
    if not substring:
        raise ValueError("substring must not be empty")
    start = text.find(substring)
    if start < 0:
        raise ValueError(f"substring {substring!r} is absent")
    if text.find(substring, start + 1) >= 0:
        raise ValueError(f"substring {substring!r} must occur exactly once")
    end = start + len(substring)

    try:
        encoded = tokenizer(text, return_offsets_mapping=True)
        try:
            offsets = encoded["offset_mapping"]
        except (KeyError, TypeError):
            offsets = encoded.offset_mapping
    except (AttributeError, NotImplementedError, TypeError) as exc:
        raise ValueError("tokenizer must provide offset mappings") from exc

    if isinstance(offsets, Tensor):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    try:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("tokenizer returned invalid offset mappings") from exc
    if not overlapping:
        raise ValueError(f"no token overlaps substring {substring!r}")
    return overlapping[-1]


def _validate_replacements(layer: int, positions: Sequence[int], states: Tensor) -> tuple[int, ...]:
    positions = tuple(positions)
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("states must have shape [n_positions, d_model]")
    if not positions:
        raise ValueError("at least one position and state are required")
    if len(positions) != states.shape[0]:
        raise ValueError("each position needs one state")
    if any(position < 0 for position in positions) or len(set(positions)) != len(positions):
        raise ValueError("positions must be unique and non-negative")
    if not bool(torch.isfinite(states).all()) or bool((states.norm(dim=1) == 0).any()):
        raise ValueError("replacement states must be finite and nonzero")
    return positions


@contextmanager
def patch_residuals(
    model: LoadedModel,
    layer: int,
    positions: Sequence[int],
    states: Tensor,
    *,
    expected_recipients: Tensor | None = None,
) -> Iterator[None]:
    """Exactly replace residual states, optionally failing on recipient drift."""
    positions = _validate_replacements(layer, positions, states)
    if not 0 <= layer < model.n_layers:
        raise ValueError(f"layer {layer} is outside [0, {model.n_layers})")
    if expected_recipients is not None and expected_recipients.shape != states.shape:
        raise ValueError("expected recipients must match replacement states")

    def patch(_module: object, _inputs: object, output: object) -> object:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError("transformer block must return [batch, sequence, hidden]")
        if states.shape[1] != hidden.shape[2] or any(
            position >= hidden.shape[1] for position in positions
        ):
            raise ValueError("replacement shape or position does not match the residual stream")
        if expected_recipients is not None:
            actual = hidden[0, list(positions)].detach().to("cpu", torch.float32)
            if not torch.equal(actual, expected_recipients.to("cpu", torch.float32)):
                raise RuntimeError("recipient residual drifted before replacement")
        edited = hidden.clone()
        edited[:, list(positions)] = states.to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    handle = model.blocks[layer].register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()
