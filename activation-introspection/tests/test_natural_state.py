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
    ARITH_DEV,
    ARITH_TEST,
    MARKER,
    NEGATIVE_BRIDGE,
    POSITIVE_BRIDGE,
    ROUTE_WORLDS,
    ArithTask,
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


def test_arithmetic_twins_differ_only_in_parity_and_one_operand() -> None:
    for bank in (ARITH_DEV, ARITH_TEST):
        assert len(bank) == 5
        answers = [task.answer(sign) for task in bank for sign in (1, -1)]
        assert sorted(answers) == list(range(10))
        for task in bank:
            assert task.answer(1) % 2 == 0
            assert task.answer(-1) % 2 == 1
            assert task.problem(1) != task.problem(-1)
            assert task.render_user(1).endswith(task.problem(1))
    dev = {task.problem(sign) for task in ARITH_DEV for sign in (1, -1)}
    held_out = {task.problem(sign) for task in ARITH_TEST for sign in (1, -1)}
    assert dev.isdisjoint(held_out)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ((6, "+", 4, 5), "single-digit"),  # 6 + 5 = 11
        ((4, "+", 5, 4), "wrong parity"),  # the twins are swapped
        ((4, "+", 4, 4), "must differ"),
        ((4, "*", 4, 5), r"\+ or -"),
    ],
)
def test_arithmetic_tasks_validate_their_own_answers(
    task: tuple[int, str, int, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArithTask(*task)


def test_layer_selection_takes_the_earliest_passing_anchor() -> None:
    def screen(*passed: bool) -> dict[int, dict[str, object]]:
        return {
            layer: {"passed": ok} for layer, ok in zip(runner.ANCHOR_LAYERS, passed, strict=True)
        }

    assert runner._select_layer(screen(False, True, True)) == runner.ANCHOR_LAYERS[1]
    assert runner._select_layer(screen(True, False, False)) == runner.ANCHOR_LAYERS[0]
    assert runner._select_layer(screen(False, False, False)) is None


def test_report_summary_preserves_paired_units_and_frozen_gates() -> None:
    rows: list[dict[str, object]] = []
    for task in ARITH_TEST:
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
                        "query_task": task.name,
                        "demo_signs": [-1, -1, 1, 1],
                        "query_sign": query_sign,
                        "positive_label": positive_label,
                        "condition_scores": {
                            "natural": score(correct),
                            "query_only": score("Q"),
                            "anti_grounded": score("K" if correct == "Q" else "Q"),
                            "clean": score("Q"),
                            "sham": score("Q"),
                            "visible": score(correct),
                        },
                    }
                )

    summary = runner._report_summary(rows)
    metrics = cast(dict[str, Any], summary["metrics"])

    assert summary["all_gates_pass"] is True
    assert summary["visible_capability_passed"] is True
    assert metrics["natural_accuracy"] == 1.0
    assert metrics["query_only_accuracy"] == 0.5
    assert metrics["visible_accuracy"] == 1.0
    assert metrics["natural_query_twin_both_correct"] == 1.0
    assert metrics["natural_mapping_flip_both_correct"] == 1.0
    assert metrics["anti_grounded_inverse_accuracy"] == 1.0
    assert set(metrics["by_task"]) == {task.name for task in ARITH_TEST}


def test_status_separates_an_uninterpretable_null_from_a_reporting_null() -> None:
    reachable: dict[str, object] = {"passed": True}
    passing: dict[str, object] = {"all_gates_pass": True}
    failed: dict[str, object] = {"all_gates_pass": False, "visible_capability_passed": True}
    blind: dict[str, object] = {"all_gates_pass": False, "visible_capability_passed": False}

    assert runner._status(None, None, None) == "stop_no_anchor_layer_reachable"
    assert runner._status(9, {"passed": False}, None) == "stop_test_bank_not_reachable"
    assert runner._status(9, reachable, passing) == "pilot_pass"
    assert runner._status(9, reachable, failed) == "stop_reporter_gate_failed"
    assert runner._status(9, reachable, blind) == "stop_reporter_gate_failed_uninterpretable"


def test_named_site_replaces_the_blind_screen_and_never_shares_its_artifacts() -> None:
    assert runner._anchors(None) == runner.ANCHOR_LAYERS
    assert runner._anchors(27) == (27,)
    assert runner._select_layer({27: {"passed": True}}, (27,)) == 27
    assert runner._select_layer({27: {"passed": False}}, (27,)) is None

    named = [
        runner._default_output(smoke=False, site=27),
        runner._default_output(smoke=True, site=27),
        runner._default_protocol(smoke=False, site=27),
        runner._default_protocol(smoke=True, site=27),
        runner._default_output(smoke=False),
        runner._default_output(smoke=True),
        runner._default_protocol(smoke=False),
        runner._default_protocol(smoke=True),
    ]
    assert len({path.name for path in named}) == len(named)
    assert named[0].name == "natural_state_arith_l27_v1_raw.jsonl"
    assert named[3].name == "natural_state_arith_l27_smoke_protocol_v1.json"

    # A post-hoc site must be recorded as one, or the artifact reads as blind.
    blind = runner._protocol(smoke=False, site=None)
    posthoc = runner._protocol(smoke=False, site=27)
    assert cast(dict[str, Any], posthoc["design"])["anchor_layers"] == [27]
    assert "post-hoc" in cast(dict[str, Any], posthoc["design"])["layer_selection"]
    assert "POST-HOC" in cast(str, posthoc["disclosed_precursor"]).upper()
    assert posthoc["report_gates"] == blind["report_gates"]
    assert posthoc["interpretation_gate"] == blind["interpretation_gate"]


def test_recovery_and_default_outputs_fail_closed() -> None:
    assert runner._normalized_recovery(-2.0, 1.0, 4.0) == 0.5
    with pytest.raises(ValueError, match="too small"):
        runner._normalized_recovery(1.0, 2.0, 1.0)
    assert runner._default_output(smoke=True).name == "natural_state_arith_smoke_v1_raw.jsonl"
    assert runner._default_output(smoke=False).name == "natural_state_arith_v1_raw.jsonl"
