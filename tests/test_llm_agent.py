"""Pure tests for the fixed-menu nuisance marginalisation."""

from __future__ import annotations

from collections import Counter

import pytest

from sandbox.llm_agent import (
    MenuObservation,
    exact_orders,
    remap_slot_probs,
    summarize_choices,
)


def test_exact_four_option_design_has_all_24_orders_once() -> None:
    orders = exact_orders(4)
    assert len(orders) == 24
    assert len(set(orders)) == 24

    placements = Counter((slot, rung) for order in orders for slot, rung in enumerate(order))
    assert set(placements.values()) == {6}


def test_slot_probabilities_are_mapped_back_to_option_identity() -> None:
    assert remap_slot_probs((2, 0, 3, 1), [0.1, 0.2, 0.3, 0.4]) == pytest.approx(
        (0.2, 0.4, 0.1, 0.3)
    )


def test_exact_order_summary_is_descriptive() -> None:
    rows = [
        MenuObservation(
            model_name="test",
            feedback_level="silent",
            reason="test",
            order=(0, 1),
            slot_probs=(0.8, 0.2),
            rung_probs=(0.8, 0.2),
        ),
        MenuObservation(
            model_name="test",
            feedback_level="silent",
            reason="test",
            order=(1, 0),
            slot_probs=(0.6, 0.4),
            rung_probs=(0.4, 0.6),
        ),
    ]
    summary = summarize_choices(rows)
    assert summary.n_orders == 2
    assert summary.probs == pytest.approx([0.6, 0.4])
    assert summary.mean_rung == pytest.approx(0.4)
