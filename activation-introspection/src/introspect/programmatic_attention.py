"""Exact sparse lowering for one released GPT-2 attention program."""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

SELF_WEIGHT = 0.01


def first_token_matrix(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the released L6H9 program: 0.99 on token 0 and 0.01 on self."""
    if length < 1:
        raise ValueError("length must be positive")
    matrix = torch.eye(length, device=device, dtype=dtype) * SELF_WEIGHT
    matrix[:, 0] += 1.0 - SELF_WEIGHT
    return matrix


def dense_first_token_mix(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply the program as a cached dense-matrix reference."""
    _validate_values(values)
    length = values.shape[-2]
    if matrix.shape != (length, length):
        raise ValueError(f"matrix must have shape ({length}, {length})")
    if matrix.device != values.device or matrix.dtype != values.dtype:
        raise ValueError("matrix and values must share device and dtype")
    return torch.matmul(matrix, values)


def sparse_first_token_mix(values: torch.Tensor) -> torch.Tensor:
    """Apply the same program without materializing its quadratic matrix."""
    _validate_values(values)
    return torch.lerp(values[..., :1, :], values, SELF_WEIGHT)


class SparseFirstTokenGPT2Attention(nn.Module):
    """Replace one GPT-2 head with its QK-free program for full-sequence inference."""

    value_weight: torch.Tensor
    value_bias: torch.Tensor
    output_weight: torch.Tensor

    def __init__(self, attention: GPT2Attention, head: int) -> None:
        super().__init__()
        if attention.is_cross_attention or attention.pruned_heads:
            raise ValueError("attention must be unpruned GPT-2 self-attention")
        if not 0 <= head < attention.num_heads:
            raise ValueError(f"head must be in [0, {attention.num_heads})")

        width = attention.head_dim
        hidden = attention.split_size
        head_slice = slice(head * width, (head + 1) * width)
        value_slice = slice(2 * hidden + head_slice.start, 2 * hidden + head_slice.stop)
        self.register_buffer(
            "value_weight", attention.c_attn.weight[:, value_slice].detach().clone()
        )
        self.register_buffer("value_bias", attention.c_attn.bias[value_slice].detach().clone())
        self.register_buffer("output_weight", attention.c_proj.weight[head_slice].detach().clone())

        attention.prune_heads({head})  # type: ignore[no-untyped-call]
        self.native = attention
        self.train(attention.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: object | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        unpadded_mask = attention_mask is None or (
            attention_mask.ndim == 4
            and attention_mask.shape[-2:] == hidden_states.shape[1:2] * 2
            and not bool(torch.count_nonzero(attention_mask[..., -1, :]))
        )
        unsupported = (
            self.training
            or past_key_values is not None
            or not unpadded_mask
            or head_mask is not None
            or encoder_hidden_states is not None
            or encoder_attention_mask is not None
            or bool(output_attentions)
            or bool(kwargs.get("use_cache", False))
        )
        if unsupported:
            raise ValueError(
                "sparse program supports eval-mode, unpadded, full-sequence self-attention only"
            )

        shape = (*hidden_states.shape[:-1], self.value_bias.numel())
        values = torch.addmm(
            self.value_bias,
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            self.value_weight,
        ).view(shape)
        program_output = torch.matmul(sparse_first_token_mix(values), self.output_weight)
        native_output, _ = self.native(
            hidden_states,
            cache_position=cache_position,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        return native_output + program_output, None


def _validate_values(values: torch.Tensor) -> None:
    if values.ndim < 2 or values.shape[-2] < 1:
        raise ValueError("values must have shape (..., sequence, head_dim) with sequence > 0")
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")
