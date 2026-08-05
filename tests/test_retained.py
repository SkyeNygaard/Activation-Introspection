"""Unit tests for the retained-trace endpoint.

The properties pinned here are the ones whose failure would silently invalidate
the result rather than raise: label-slot balance, the guarantee that no codebook
token is in the carrier stage, and the arm definitions that make `clean` and
`sham` genuine chance floors.
"""

from __future__ import annotations

import pytest
import torch

from introspect.concepts import ConceptVector
from introspect.retained import (
    DEV_CONCEPTS,
    SINK_SKIP,
    TEST_CONCEPTS,
    Codebook,
    balanced_codebooks,
    build_arm,
    carrier_kl,
    carrier_positions,
)


def test_concept_banks_are_disjoint() -> None:
    """Tuning on DEV and reporting on TEST only means something if they differ."""
    assert not set(DEV_CONCEPTS) & set(TEST_CONCEPTS)
    assert len(DEV_CONCEPTS) == len(TEST_CONCEPTS) == 8


def test_balanced_codebooks_put_each_concept_in_each_slot_once() -> None:
    concepts = ["a", "b", "c", "d"]
    labels = ["Q", "K", "Z", "J"]
    books = balanced_codebooks(concepts, labels)
    assert len(books) == 4
    for concept in concepts:
        assigned = [cb.mapping[concept] for cb in books]
        # Exact balance is what makes chance exactly 1/n. A random draw would
        # leave a slot bias indistinguishable from a real effect.
        assert sorted(assigned) == sorted(labels)
    for cb in books:
        assert sorted(cb.labels) == sorted(labels)
        assert sorted(cb.order) == sorted(concepts)


def test_balanced_codebooks_decorrelate_display_order_from_labels() -> None:
    """Line position must not be a proxy for label, or one artifact fakes both."""
    concepts = ["a", "b", "c", "d", "e", "f", "g", "h"]
    labels = ["Q", "K", "Z", "J", "X", "V", "W", "Y"]
    books = balanced_codebooks(concepts, labels)
    orders = {cb.order for cb in books}
    assert len(orders) > 1, "display order never varies"
    # 'a' must not always be rendered on the same line.
    positions = {cb.order.index("a") for cb in books}
    assert len(positions) > 1


def test_balanced_codebooks_rejects_short_label_alphabet() -> None:
    with pytest.raises(ValueError, match="need 3 labels"):
        balanced_codebooks(["a", "b", "c"], ["Q", "K"])


def test_codebook_digest_tracks_assignment_not_dict_insertion() -> None:
    order = ("x", "y")
    a = Codebook({"x": "Q", "y": "K"}, order)
    b = Codebook({"y": "K", "x": "Q"}, order)
    assert a.digest() == b.digest()
    assert a.digest() != Codebook({"x": "K", "y": "Q"}, order).digest()


def test_clean_arm_registers_nothing_and_sham_is_a_zero_strength_edit() -> None:
    vec = ConceptVector(name="v", layer=3, vector=torch.ones(8))
    positions = [1, 2, 3]

    assert build_arm("clean", vec, 3, 2.0, positions) == []

    sham = build_arm("sham", vec, 3, 2.0, positions)
    assert len(sham) == 1
    assert sham[0].strength == 0.0
    # Same hook, same positions, same layer as a real arm: the pair separates
    # "an edit happened" from "a hook ran".
    assert sham[0].layer == 3
    assert list(sham[0].positions) == positions

    target = build_arm("target", vec, 3, 2.0, positions)
    assert target[0].strength == 2.0


def test_sham_edit_is_numerically_a_no_op() -> None:
    vec = ConceptVector(name="v", layer=0, vector=torch.randn(8))
    iv = build_arm("sham", vec, 0, 5.0, [0, 1])[0]
    hidden = torch.randn(1, 3, 8)
    mask = torch.tensor([True, True, False])
    assert torch.allclose(iv.apply(hidden, mask), hidden)


def test_carrier_positions_skips_the_attention_sink() -> None:
    positions = carrier_positions(20)
    assert positions[0] == SINK_SKIP
    assert positions[-1] == 19
    # Position 0 holds an activation orders of magnitude above a normal
    # residual; editing there swamps the concept direction.
    assert 0 not in positions


def test_carrier_positions_rejects_a_carrier_shorter_than_the_sink() -> None:
    with pytest.raises(ValueError, match="need more than"):
        carrier_positions(SINK_SKIP)


def test_carrier_kl_is_zero_for_identical_logits_and_positive_otherwise() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    assert carrier_kl(a, a) == pytest.approx(0.0, abs=1e-6)
    assert carrier_kl(a, torch.tensor([3.0, 2.0, 1.0])) > 0.0
