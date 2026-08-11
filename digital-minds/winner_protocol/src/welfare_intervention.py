"""Targeted ActAdd matching the published welfare steering convention.

This intentionally does NOT reuse introspect.hooks.Intervention:
- no unit-normalization of the concept vector;
- no residual-norm scaling;
- installs a forward PRE-hook (block input);
- edits only explicitly named token positions.

The source welfare code applies `factor * vector` at a chosen block input.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

@dataclass(frozen=True)
class PublishedActAdd:
    layer: int
    vector: Tensor
    factor: float
    positions: tuple[int, ...]
    signs: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.signs):
            raise ValueError("each position needs one sign")
        if any(s not in (-1, +1) for s in self.signs):
            raise ValueError("signs must be +/-1")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        if self.vector.ndim != 1:
            raise ValueError("vector must be [d_model]")

def _apply(hidden: Tensor, edit: PublishedActAdd) -> Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, seq, d], got {tuple(hidden.shape)}")
    if hidden.shape[0] != 1:
        raise ValueError("confirmation runner is deliberately single-example")
    out = hidden.clone()
    vec = edit.vector.to(device=hidden.device, dtype=hidden.dtype)
    if vec.numel() != hidden.shape[-1]:
        raise ValueError("vector width does not match residual width")
    for pos, sign in zip(edit.positions, edit.signs, strict=True):
        if not 0 <= pos < hidden.shape[1]:
            raise IndexError(f"position {pos} outside sequence length {hidden.shape[1]}")
        out[:, pos, :] = out[:, pos, :] + (sign * edit.factor) * vec
    return out

@contextmanager
def targeted_published_actadd(blocks, edit: PublishedActAdd) -> Iterator[None]:
    """Install one pre-hook and guarantee cleanup."""
    if not 0 <= edit.layer < len(blocks):
        raise IndexError("layer outside model")

    def hook(_module, args):
        if not args:
            raise ValueError("transformer block pre-hook received no inputs")
        hidden = args[0]
        if not isinstance(hidden, Tensor):
            raise TypeError("first block input is not a tensor")
        return (_apply(hidden, edit), *args[1:])

    handle: RemovableHandle = blocks[edit.layer].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def load_hf_vector(
    *,
    repo_id: str,
    filename: str,
    layer: int,
    position: int = 0,
    revision: str | None = None,
) -> Tensor:
    """Download and select one `(position, layer, :)` vector externally.

    This sandbox has no network/model weights; the function is for the GPU runner.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 3:
        raise ValueError(f"expected (positions,layers,d), got {tuple(tensor.shape)}")
    if not 0 <= position < tensor.shape[0] or not 0 <= layer < tensor.shape[1]:
        raise IndexError("position/layer outside vector tensor")
    return tensor[position, layer].float().contiguous()
