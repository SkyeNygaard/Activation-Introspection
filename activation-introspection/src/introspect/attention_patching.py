"""Causal path patching at per-head attention outputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from introspect.models import LoadedModel


def _output_projection(model: LoadedModel, layer: int) -> nn.Module:
    if not 0 <= layer < model.n_layers:
        raise ValueError(f"layer {layer} is outside [0, {model.n_layers})")
    attention = getattr(model.blocks[layer], "self_attn", None)
    projection = getattr(attention, "o_proj", None)
    if not isinstance(projection, nn.Module):
        raise ValueError(f"layer {layer} does not expose self_attn.o_proj")
    return projection


@dataclass
class AttentionInputs:
    """Full pre-output-projection attention contexts."""

    by_layer: dict[int, Tensor] = field(default_factory=dict)


@contextmanager
def capture_attention_inputs(
    model: LoadedModel, layers: Sequence[int]
) -> Iterator[AttentionInputs]:
    store = AttentionInputs()
    handles: list[RemovableHandle] = []

    def make(layer: int):  # type: ignore[no-untyped-def]
        def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            if layer in store.by_layer:
                raise RuntimeError(f"layer {layer} attention ran more than once")
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise ValueError("attention output projection must receive [batch, seq, hidden]")
            store.by_layer[layer] = inputs[0].detach().to("cpu", torch.float32)

        return capture

    try:
        for layer in layers:
            handles.append(_output_projection(model, layer).register_forward_pre_hook(make(layer)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_attention_inputs(
    model: LoadedModel,
    donors: Mapping[int, Tensor],
    heads_by_layer: Mapping[int, Sequence[int]],
    *,
    n_heads: int,
    expected_recipients: Mapping[int, Tensor] | None = None,
    positions: Literal["all"] | Sequence[int] = (-1,),
) -> Iterator[None]:
    """Replace selected head contexts at chosen positions with donors."""
    if n_heads < 1:
        raise ValueError("n_heads must be positive")
    handles: list[RemovableHandle] = []

    def make(layer: int, heads: tuple[int, ...]):  # type: ignore[no-untyped-def]
        def patch(_module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor]:
            if len(inputs) != 1 or inputs[0].ndim != 3:
                raise ValueError("attention output projection must receive [batch, seq, hidden]")
            hidden = inputs[0]
            donor = donors[layer]
            if hidden.shape[-1] % n_heads:
                raise ValueError("attention width must divide evenly into query heads")
            if donor.shape != hidden.shape:
                raise ValueError("donor must match the recipient batch, sequence, and hidden width")
            if expected_recipients is not None:
                expected = expected_recipients[layer]
                actual = hidden.detach().to("cpu", torch.float32)
                if not torch.equal(actual, expected):
                    raise ValueError(f"recipient context drifted before patch at layer {layer}")
            width = hidden.shape[-1] // n_heads
            edited = hidden.clone()
            selected = list(range(hidden.shape[1])) if positions == "all" else list(positions)
            if not selected or any(
                not -hidden.shape[1] <= position < hidden.shape[1] for position in selected
            ):
                raise ValueError("patch positions must be nonempty and inside the sequence")
            for head in heads:
                if not 0 <= head < n_heads:
                    raise ValueError(f"head {head} is outside [0, {n_heads})")
                start = head * width
                edited[:, selected, start : start + width] = donor[
                    :, selected, start : start + width
                ].to(hidden.device, hidden.dtype)
            return (edited,)

        return patch

    try:
        for layer, requested in heads_by_layer.items():
            heads = tuple(sorted(set(requested)))
            if layer not in donors:
                raise ValueError(f"missing donor for layer {layer}")
            if not heads:
                raise ValueError(f"layer {layer} has no heads to patch")
            handles.append(
                _output_projection(model, layer).register_forward_pre_hook(make(layer, heads))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
