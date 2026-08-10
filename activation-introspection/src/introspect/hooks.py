"""Read and write the residual stream with forward hooks.

Two primitives:

- ``capture`` records the residual stream at chosen layers (used to build concept
  vectors from contrast pairs).
- ``intervene`` adds, replaces, or ablates a direction at chosen layers and token
  positions while the model runs.

Both are context managers so a hook can never outlive the block it was measuring,
which is the usual source of silent contamination between conditions.

Nesting order is semantic. Forward hooks fire in registration order, so::

    with intervene(model, ivs), capture(model, layers) as store:   # post-edit
    with capture(model, layers) as store, intervene(model, ivs):   # pre-edit

Getting this backwards records clean activations under an "intervened" label and
nothing ever errors. ``tests/test_hooks.py`` pins both directions.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from introspect.models import LoadedModel

Positions = Literal["all", "last", "generated"] | Sequence[int]
Mode = Literal["add", "replace", "ablate"]


def _hidden(output: object) -> Tensor:
    """Blocks return either a Tensor or a tuple whose first element is one."""
    if isinstance(output, tuple):
        return cast(Tensor, output[0])
    assert isinstance(output, Tensor)
    return output


def _rewrap(output: object, new: Tensor) -> object:
    if isinstance(output, tuple):
        return (new, *output[1:])
    return new


def _position_mask(seq_len: int, positions: Positions, prompt_len: int) -> Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if positions == "all":
        mask[:] = True
    elif positions == "last":
        mask[-1] = True
    elif positions == "generated":
        # During generation with a KV cache, forward passes after the first carry
        # exactly one token, so "everything past the prompt" reduces to this.
        if seq_len == 1:
            mask[0] = True
        else:
            mask[prompt_len:] = True
    else:
        for i in positions:
            mask[i] = True
    return mask


@dataclass(frozen=True)
class Intervention:
    """A single steering edit applied to the residual stream.

    ``strength`` is interpreted relative to the *measured* norm of the residual
    stream at that layer when ``normalize`` is true. Absolute magnitudes are not
    comparable across layers -- residual norm grows with depth -- and reporting a
    raw alpha across a layer sweep is a standard way to produce a fake result.
    """

    layer: int
    direction: Tensor  # [d_model], not required to be unit norm
    strength: float = 1.0
    positions: Positions = "all"
    mode: Mode = "add"
    normalize: bool = True
    per_position: bool = False
    label: str = ""

    def apply(self, hidden: Tensor, mask: Tensor) -> Tensor:
        vec = self.direction.to(hidden.device, hidden.dtype)
        unit = vec / (vec.norm() + 1e-8)
        selected = hidden[:, mask, :]

        if self.mode == "ablate":
            proj = (selected @ unit).unsqueeze(-1) * unit
            edited = selected - proj
        else:
            scale = self.strength
            if self.normalize:
                # Match the typical residual magnitude at this layer and position.
                norms = selected.norm(dim=-1, keepdim=True)
                scale = self.strength * (norms if self.per_position else norms.mean().item())
            delta = scale * unit
            edited = delta.expand_as(selected) if self.mode == "replace" else selected + delta

        out = hidden.clone()
        out[:, mask, :] = edited.to(hidden.dtype)
        return out


@dataclass
class Capture:
    """Collected residual-stream activations, keyed by layer index."""

    acts: dict[int, list[Tensor]] = field(default_factory=dict)

    def add(self, layer: int, value: Tensor) -> None:
        self.acts.setdefault(layer, []).append(value.detach().to("cpu", torch.float32))

    def last_token(self, layer: int) -> Tensor:
        """[batch, d_model] at the final position of the first recorded pass."""
        return self.acts[layer][0][:, -1, :]


@contextmanager
def capture(model: LoadedModel, layers: Sequence[int]) -> Iterator[Capture]:
    store = Capture()
    handles: list[RemovableHandle] = []
    blocks = model.blocks

    def make(layer: int):  # type: ignore[no-untyped-def]
        def hook(_mod: object, _inp: object, output: object) -> None:
            store.add(layer, _hidden(output))

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make(layer)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def intervene(
    model: LoadedModel,
    interventions: Sequence[Intervention],
    *,
    prompt_len: int = 0,
) -> Iterator[None]:
    """Apply interventions for the duration of the block.

    ``prompt_len`` is only consulted by ``positions="generated"``.
    """
    handles: list[RemovableHandle] = []
    blocks = model.blocks
    by_layer: dict[int, list[Intervention]] = {}
    for iv in interventions:
        by_layer.setdefault(iv.layer, []).append(iv)

    def make(layer_ivs: list[Intervention]):  # type: ignore[no-untyped-def]
        def hook(_mod: object, _inp: object, output: object) -> object:
            hidden = _hidden(output)
            seq_len = hidden.shape[1]
            for iv in layer_ivs:
                mask = _position_mask(seq_len, iv.positions, prompt_len).to(hidden.device)
                hidden = iv.apply(hidden, mask)
            return _rewrap(output, hidden)

        return hook

    try:
        for layer, ivs in by_layer.items():
            handles.append(blocks[layer].register_forward_hook(make(ivs)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate(
    model: LoadedModel,
    prompt: str,
    *,
    interventions: Sequence[Intervention] = (),
    max_new_tokens: int = 48,
    temperature: float = 0.0,
) -> str:
    """Greedy (or sampled) continuation, optionally under intervention."""
    ids = model.encode(prompt)
    prompt_len = int(ids.shape[1])
    kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": model.tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature

    with intervene(model, interventions, prompt_len=prompt_len):
        out = model.generate_ids(ids, **kwargs)

    return str(model.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))
