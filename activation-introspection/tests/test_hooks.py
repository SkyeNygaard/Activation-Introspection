"""Unit tests for the intervention primitives, using a stub model.

These deliberately avoid downloading weights so the mechanics can be tested in a
second. The one thing they must guarantee is that an intervention touches exactly
the positions it claims to and nothing else -- a position-mask bug silently turns
"injected during generation only" into "injected everywhere", which would make an
introspection result meaningless.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from introspect.hooks import Intervention, _position_mask, capture, intervene
from introspect.models import LoadedModel


class _Block(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + 1.0,)


class _Stub(nn.Module):
    def __init__(self, n_layers: int = 3, d_model: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.config = type("C", (), {"hidden_size": d_model})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in cast(nn.ModuleList, self.model.layers):
            x = blk(x)[0]
        return x


def _loaded() -> LoadedModel:
    stub = _Stub()
    return LoadedModel(
        name="stub",
        model=stub,
        tokenizer=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_position_mask_variants() -> None:
    assert _position_mask(4, "all", 0).tolist() == [True] * 4
    assert _position_mask(4, "last", 0).tolist() == [False, False, False, True]
    assert _position_mask(4, "generated", 2).tolist() == [False, False, True, True]
    # Single-token forward during cached generation counts as generated.
    assert _position_mask(1, "generated", 7).tolist() == [True]
    assert _position_mask(3, [0, 2], 0).tolist() == [True, False, True]


def test_add_touches_only_masked_positions() -> None:
    lm = _loaded()
    x = torch.zeros(1, 3, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    iv = Intervention(layer=0, direction=direction, strength=5.0, positions=[1], normalize=False)

    with intervene(lm, [iv]):
        out = lm.model(x)

    baseline = lm.model(x)
    delta = out - baseline
    assert delta[0, 0].abs().sum() == 0
    assert delta[0, 2].abs().sum() == 0
    assert torch.allclose(delta[0, 1], torch.tensor([5.0, 0.0, 0.0, 0.0]))


def test_ablate_removes_the_component() -> None:
    lm = _loaded()
    x = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]])
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    iv = Intervention(layer=0, direction=direction, mode="ablate")

    # Register the intervention FIRST so the capture hook observes the edited
    # stream. See test_capture_order_decides_what_is_recorded.
    with intervene(lm, [iv]), capture(lm, [0]) as store:
        lm.model(x)

    # Block 0 adds 1 before the hooks run, so pre-edit is [4,5,1,1];
    # ablating the first coordinate must zero it.
    edited = store.acts[0][0]
    assert edited[0, 0, 0].abs() < 1e-6
    assert torch.allclose(edited[0, 0, 1:], torch.tensor([5.0, 1.0, 1.0]))


def test_capture_order_decides_what_is_recorded() -> None:
    """Forward hooks fire in registration order, so nesting order is semantic.

    Pinning this because getting it backwards is silent: you record the clean
    activations, label them "intervened", and every downstream number is wrong
    without anything ever erroring.
    """
    lm = _loaded()
    x = torch.zeros(1, 1, 4)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    iv = Intervention(layer=0, direction=direction, strength=9.0, normalize=False)

    with capture(lm, [0]) as before, intervene(lm, [iv]):
        lm.model(x)
    with intervene(lm, [iv]), capture(lm, [0]) as after:
        lm.model(x)

    assert before.acts[0][0][0, 0, 0].item() == 1.0  # pre-edit
    assert after.acts[0][0][0, 0, 0].item() == 10.0  # post-edit


def test_hooks_are_removed_on_exit() -> None:
    lm = _loaded()
    x = torch.zeros(1, 2, 4)
    iv = Intervention(layer=1, direction=torch.ones(4), strength=3.0, normalize=False)

    with intervene(lm, [iv]):
        pass

    assert torch.allclose(lm.model(x), torch.full((1, 2, 4), 3.0))


def test_normalize_scales_with_residual_magnitude() -> None:
    lm = _loaded()
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    iv = Intervention(layer=0, direction=direction, strength=1.0, normalize=True)

    small = torch.zeros(1, 1, 4)
    large = torch.full((1, 1, 4), 100.0)
    with intervene(lm, [iv]):
        d_small = (lm.model(small) - lm.model(small.clone()) * 0).clone()
    with intervene(lm, [iv]):
        d_large = lm.model(large)

    # The injected component should be far larger where the residual is larger.
    assert d_large[0, 0, 0] > d_small[0, 0, 0] * 10
