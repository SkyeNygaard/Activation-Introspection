"""Checks for the zero-demonstration reporting task.

The load-bearing property is that visible text carries no sign information, so
these tests pin the twin structure and the scoring sign convention rather than
exercising a model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from introspect.report_training import (
    CENTERING_CONCEPTS,
    EVAL_CARRIERS,
    EVAL_CONCEPTS,
    LABELS,
    TRAIN_CARRIERS,
    TRAIN_CONCEPTS,
    PreparedReport,
    label_for,
    render_user,
    score_logits,
    sign_intervention,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_report_training as analysis


def _prepared() -> PreparedReport:
    return PreparedReport(
        carrier="x",
        prompt="p",
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        marker_position=3,
        label_ids=(10, 11),
    )


def test_twin_members_have_byte_identical_prompts() -> None:
    """The whole design rests on this: sign changes nothing visible."""
    for carrier in EVAL_CARRIERS:
        assert render_user(carrier) == render_user(carrier)
        assert label_for(1) != label_for(-1)
        # Neither label appears in the observation, so the text cannot hint.
        observation = render_user(carrier).rsplit("Observation:", 1)[1]
        for label in LABELS:
            assert label not in observation


def test_banks_are_disjoint() -> None:
    train, evaluate, centering = set(TRAIN_CONCEPTS), set(EVAL_CONCEPTS), set(CENTERING_CONCEPTS)
    assert not train & evaluate
    assert not train & centering
    assert not evaluate & centering
    assert not set(TRAIN_CARRIERS) & set(EVAL_CARRIERS)


def test_score_follows_the_sign_not_the_label() -> None:
    prepared = _prepared()
    logits = torch.zeros(1, 1, 32)
    logits[0, -1, 10] = 5.0  # favours Q, the +1 label

    positive = score_logits(logits, prepared, sign=1)
    negative = score_logits(logits, prepared, sign=-1)

    assert positive.correct is True
    assert negative.correct is False
    assert positive.signed_margin == pytest.approx(-negative.signed_margin)
    assert positive.format_ok is True


def test_clean_arm_has_no_correct_label() -> None:
    score = score_logits(torch.zeros(1, 1, 32), _prepared(), sign=0)
    assert score.correct is None
    assert score.correct_probability is None


def test_clean_arm_makes_no_edit() -> None:
    assert sign_intervention(torch.ones(4), 9, 3, 0, strength=1.0, label="x") == []
    edits = sign_intervention(torch.ones(4), 9, 3, -1, strength=1.0, label="x")
    assert len(edits) == 1
    assert torch.equal(edits[0].direction, -torch.ones(4))
    assert edits[0].positions == [3]


def _row(arm: str, condition: str, concept: str, sign: int, correct: bool) -> dict[str, object]:
    return {
        "arm": arm,
        "condition": condition,
        "concept": concept,
        "carrier_sha256": "aaaaaaaa",
        "sign": sign,
        "correct": correct,
        "signed_margin": 1.0 if correct else -1.0,
        "label_mass": 0.99,
        "format_ok": True,
        "predicted_label": "Q",
    }


def test_twin_pair_accuracy_punishes_a_constant_answerer() -> None:
    """A model that always says Q is at 0.5 per row but 0.0 on pairs.

    This is the reason the estimand is the pair, not the row.
    """
    rows = [_row("trained", "target", "a", sign, correct=(sign == 1)) for sign in (-1, 1)] + [
        _row("trained", "target", "b", sign, correct=(sign == 1)) for sign in (-1, 1)
    ]

    result = analysis._arm_condition(rows, "trained", "target")
    assert result["row_accuracy"] == 0.5
    assert result["twin_pair_accuracy"] == 0.0


def test_analyzer_rejects_an_incomplete_twin() -> None:
    with pytest.raises(SystemExit):
        analysis._arm_condition([_row("trained", "target", "a", 1, True)], "trained", "target")
