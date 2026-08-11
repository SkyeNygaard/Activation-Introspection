from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from introspect.models import LoadedModel
from introspect.natural_state import (
    MARKER,
    NEGATIVE_BRIDGE,
    POSITIVE_BRIDGE,
    ROUTE_WORLDS,
    _validate_replacements,
    patch_residuals,
    unique_substring_token_position,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_natural_state as runner


def test_route_twins_differ_only_in_selected_bridge() -> None:
    expected = "\n".join(
        [
            "Follow exactly two arrows from Start. Reply with only the final destination.",
            "sable -> maple",
            "maple -> amber",
            "cedar -> violet",
            "Start: sable",
            f"Hidden state marker: {MARKER}",
        ]
    )
    assert ROUTE_WORLDS[0].render_user(POSITIVE_BRIDGE) == expected

    for world in ROUTE_WORLDS:
        positive = world.render_user(POSITIVE_BRIDGE)
        negative = world.render_user(NEGATIVE_BRIDGE)
        assert (
            positive.replace(
                f"{world.start} -> {POSITIVE_BRIDGE}",
                f"{world.start} -> {NEGATIVE_BRIDGE}",
                1,
            )
            == negative
        )
        assert world.endpoint(POSITIVE_BRIDGE) in positive
        assert world.endpoint(NEGATIVE_BRIDGE) in positive
        assert positive.count(MARKER) == negative.count(MARKER) == 1


def test_five_worlds_have_unique_starts_and_endpoints() -> None:
    assert len(ROUTE_WORLDS) == 5
    assert len({world.start for world in ROUTE_WORLDS}) == 5
    endpoints = {
        endpoint
        for world in ROUTE_WORLDS
        for endpoint in (world.positive_endpoint, world.negative_endpoint)
    }
    assert len(endpoints) == 10


class _OffsetTokenizer:
    def __call__(self, _text: str, **_kwargs: object) -> SimpleNamespace:
        # Arbitrary multi-character token chunks plus special-token offsets.
        return SimpleNamespace(offset_mapping=torch.tensor([[[0, 0], [0, 4], [4, 9], [9, 20]]]))


def test_marker_location_uses_offsets_not_token_ids() -> None:
    tokenizer = _OffsetTokenizer()
    encoded = tokenizer("irrelevant")
    position = unique_substring_token_position(tokenizer, "abcdTARGETsuffix", "TARGET")

    assert position == 3
    assert not hasattr(encoded, "input_ids")


@pytest.mark.parametrize("text, substring", [("plain", "missing"), ("x x", "x"), ("x", "")])
def test_marker_location_requires_one_nonempty_occurrence(text: str, substring: str) -> None:
    with pytest.raises(ValueError):
        unique_substring_token_position(_OffsetTokenizer(), text, substring)


class _Block(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value + 1,)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _Stub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(tuple[torch.Tensor], self.model.layers[0](value))[0]


def _loaded_stub() -> LoadedModel:
    return LoadedModel(
        name="stub",
        model=_Stub(),
        tokenizer=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_patch_residuals_is_exact_guarded_and_removed() -> None:
    model = _loaded_stub()
    inputs = torch.zeros(1, 3, 2)
    expected = torch.ones(1, 2)
    replacement = torch.tensor([[3.0, 4.0]])

    with patch_residuals(model, 0, [1], replacement, expected_recipients=expected):
        patched = model.model(inputs)

    assert torch.equal(patched[0, 1], replacement[0])
    assert torch.equal(patched[0, [0, 2]], torch.ones(2, 2))
    assert torch.equal(model.model(inputs), torch.ones_like(inputs))
    with pytest.raises(RuntimeError, match="drifted"):
        with patch_residuals(model, 0, [1], replacement, expected_recipients=torch.zeros(1, 2)):
            model.model(inputs)


@pytest.mark.parametrize(
    ("layer", "positions", "states", "message"),
    [
        (9, [], torch.empty(0, 2), "at least one"),
        (9, [1], torch.ones(2, 2), "one state"),
        (9, [1], torch.ones(2), "shape"),
        (9, [1], torch.zeros(1, 2), "nonzero"),
        (9, [1, 1], torch.ones(2, 2), "unique"),
        (-1, [1], torch.ones(1, 2), "non-negative"),
    ],
)
def test_replacements_validate_inputs(
    layer: int, positions: list[int], states: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_replacements(layer, positions, states)


def test_report_summary_preserves_paired_units_and_frozen_gates() -> None:
    rows: list[dict[str, object]] = []
    for world in ROUTE_WORLDS:
        for positive_label in ("Q", "K"):
            negative_label = "K" if positive_label == "Q" else "Q"
            for query_sign in (-1, 1):
                correct = positive_label if query_sign == 1 else negative_label

                def score(predicted: str, expected: str = correct) -> dict[str, object]:
                    return {
                        "predicted": predicted,
                        "correct_target": predicted == expected,
                        "correct_inverse": predicted != expected,
                        "format_ok": True,
                        "label_mass": 1.0,
                        "full_logprobs": {"Q": -0.5, "K": -0.5},
                    }

                rows.append(
                    {
                        "query_world": world.start,
                        "demo_signs": [-1, -1, 1, 1],
                        "query_sign": query_sign,
                        "positive_label": positive_label,
                        "condition_scores": {
                            "natural": score(correct),
                            "query_only": score("Q"),
                            "anti_grounded": score("K" if correct == "Q" else "Q"),
                            "clean": score("Q"),
                            "sham": score("Q"),
                        },
                    }
                )

    summary = runner._report_summary(rows)
    metrics = cast(dict[str, Any], summary["metrics"])

    assert summary["all_gates_pass"] is True
    assert metrics["natural_accuracy"] == 1.0
    assert metrics["query_only_accuracy"] == 0.5
    assert metrics["natural_query_twin_both_correct"] == 1.0
    assert metrics["natural_mapping_flip_both_correct"] == 1.0
    assert metrics["anti_grounded_inverse_accuracy"] == 1.0


def test_recovery_and_default_outputs_fail_closed() -> None:
    assert runner._normalized_recovery(-2.0, 1.0, 4.0) == 0.5
    with pytest.raises(ValueError, match="too small"):
        runner._normalized_recovery(1.0, 2.0, 1.0)
    assert runner._default_output(smoke=True).name == "natural_state_smoke_v3_raw.jsonl"
    assert runner._default_output(smoke=False).name == "natural_state_dev_v1_raw.jsonl"
