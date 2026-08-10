from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from introspect.attention_patching import capture_attention_inputs, patch_attention_inputs
from introspect.models import LoadedModel


class _Attention(nn.Module):
    o_proj: nn.Identity

    def __init__(self) -> None:
        super().__init__()
        self.o_proj = nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.o_proj(hidden))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.self_attn(hidden),)


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Body()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return cast(tuple[torch.Tensor], self.model.layers[0](hidden))[0]


def test_attention_patch_changes_only_requested_answer_head_and_cleans_up() -> None:
    network = _Model()
    loaded = LoadedModel(
        name="stub",
        model=network,
        tokenizer=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    hidden = torch.arange(12, dtype=torch.float32).view(1, 3, 4)

    with capture_attention_inputs(loaded, [0]) as captured:
        baseline = network(hidden)
    assert torch.equal(captured.by_layer[0], hidden)

    donor = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    donor_context = hidden.clone()
    donor_context[:, -1] = donor
    with patch_attention_inputs(
        loaded,
        {0: donor_context},
        {0: [1]},
        n_heads=2,
        expected_recipients=captured.by_layer,
        positions=[-1],
    ):
        patched = network(hidden)

    expected = baseline.clone()
    expected[:, -1, 2:] = donor[:, 2:]
    assert torch.equal(patched, expected)
    assert torch.equal(network(hidden), baseline)

    with pytest.raises(ValueError, match="layer -1 is outside"):
        with capture_attention_inputs(loaded, [-1]):
            pass
