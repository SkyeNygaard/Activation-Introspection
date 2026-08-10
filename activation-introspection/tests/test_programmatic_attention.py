from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from introspect.programmatic_attention import (
    SparseFirstTokenGPT2Attention,
    dense_first_token_mix,
    first_token_matrix,
    sparse_first_token_mix,
)


def test_released_first_token_program_lowers_without_a_matrix() -> None:
    values = torch.tensor(
        [[[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]
    )
    matrix = first_token_matrix(3, device=values.device, dtype=values.dtype)

    assert torch.equal(matrix.sum(dim=-1), torch.ones(3))
    assert torch.equal(matrix.triu(diagonal=1), torch.zeros_like(matrix))
    assert torch.count_nonzero(matrix).item() == 5
    assert torch.allclose(dense_first_token_mix(values, matrix), sparse_first_token_mix(values))

    singleton = values[:, :1]
    assert torch.equal(sparse_first_token_mix(singleton), singleton)
    with pytest.raises(ValueError, match="sequence > 0"):
        sparse_first_token_mix(values[:, :0])


def test_gpt2_lowering_prunes_qk_and_preserves_dense_program_output() -> None:
    torch.manual_seed(4)
    config = GPT2Config(  # type: ignore[no-untyped-call]
        n_embd=96,
        n_head=12,
        n_layer=12,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    config._attn_implementation = "eager"
    original = GPT2Attention(config, layer_idx=6).eval()  # type: ignore[no-untyped-call]
    dense = copy.deepcopy(original)
    sparse = SparseFirstTokenGPT2Attention(copy.deepcopy(original), head=9).eval()
    hidden = torch.randn(2, 17, config.n_embd)

    captured: dict[str, torch.Tensor] = {}
    head_slice = slice(9 * 8, 10 * 8)
    value_slice = slice(2 * config.n_embd + head_slice.start, 2 * config.n_embd + head_slice.stop)

    def capture_values(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured["values"] = output[..., value_slice]

    def replace_head(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor]:
        contexts = inputs[0].clone()
        matrix = first_token_matrix(17, device=hidden.device, dtype=hidden.dtype)
        contexts[..., head_slice] = dense_first_token_mix(captured["values"], matrix)
        return (contexts,)

    capture_handle = dense.c_attn.register_forward_hook(capture_values)
    replace_handle = dense.c_proj.register_forward_pre_hook(replace_head)
    try:
        with torch.inference_mode():
            expected = dense(hidden)[0]
            actual = sparse(hidden)[0]
    finally:
        capture_handle.remove()
        replace_handle.remove()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert sparse.native.num_heads == 11
    assert sparse.native.c_attn.weight.shape == (96, 264)
    assert sparse.native.c_proj.weight.shape == (88, 96)
    with pytest.raises(ValueError, match="full-sequence"):
        sparse(hidden, past_key_values=object())

    model_config = GPT2Config(  # type: ignore[no-untyped-call]
        vocab_size=32,
        n_positions=16,
        n_embd=96,
        n_head=12,
        n_layer=7,
        use_cache=False,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model_config._attn_implementation = "eager"
    model = GPT2LMHeadModel(model_config).eval()  # type: ignore[no-untyped-call]
    model.transformer.h[6].attn = SparseFirstTokenGPT2Attention(
        model.transformer.h[6].attn, head=9
    ).eval()
    with torch.inference_mode():
        logits = model(torch.arange(5).unsqueeze(0), use_cache=False).logits
    assert logits.shape == (1, 5, 32)
