"""Tests for the statistics that decide whether a reported gap is real."""

from __future__ import annotations

import math

import pytest
import torch

from introspect.grading import ChoiceResult, grade_free_form
from introspect.metrics import (
    accuracy,
    auroc,
    bootstrap,
    bootstrap_auroc,
    kl_from_clean,
    paired_difference,
)


def test_auroc_corrects_for_ties() -> None:
    """Detection answers take few discrete values; ties are the common case."""
    assert auroc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert auroc([1.0], [0.0]) == pytest.approx(1.0)
    assert auroc([0.0], [1.0]) == pytest.approx(0.0)
    # Half the positives tie with the negative: 0.5 credit for the tie.
    assert auroc([1.0, 0.0], [0.0]) == pytest.approx(0.75)


def test_auroc_is_degenerate_when_the_null_has_no_variance() -> None:
    """The failure that motivated prompt paraphrases.

    With one fixed prompt and greedy decoding every clean trial is identical, so
    the null distribution is a point mass and AUROC can only be 0, 0.5, or 1 --
    it carries no information about the model.
    """
    constant_null = [0.7] * 10
    assert auroc([0.6] * 10, constant_null) == pytest.approx(0.0)
    assert auroc([0.8] * 10, constant_null) == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_mean() -> None:
    est = bootstrap([0.0, 1.0] * 50, seed=0)
    assert est.value == pytest.approx(0.5)
    assert est.lo < 0.5 < est.hi
    assert est.n == 100


def test_bootstrap_interval_narrows_with_more_data() -> None:
    small = bootstrap([0.0, 1.0] * 5, seed=0)
    large = bootstrap([0.0, 1.0] * 500, seed=0)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_estimate_excludes_zero_only_when_it_should() -> None:
    assert bootstrap([1.0] * 20).excludes
    assert not bootstrap([-1.0, 1.0] * 20).excludes


def test_paired_difference_requires_alignment() -> None:
    """Misaligned arms would silently compare different trials."""
    with pytest.raises(ValueError, match="equal length"):
        paired_difference([1.0, 0.0], [1.0])


def test_paired_difference_is_tighter_than_unpaired() -> None:
    """Why the gap is computed paired: both arms see the same trial."""
    a = [1.0, 0.0] * 25
    b = [1.0, 0.0] * 25  # perfectly correlated with a
    paired = paired_difference(a, b)
    assert paired.value == pytest.approx(0.0)
    # Perfect correlation means the difference is identically zero.
    assert paired.hi - paired.lo == pytest.approx(0.0)


def test_bootstrap_auroc_covers_a_real_separation() -> None:
    est = bootstrap_auroc([0.9] * 30, [0.1] * 30)
    assert est.value == pytest.approx(1.0)
    assert est.lo > 0.9


def test_kl_is_zero_for_identical_distributions() -> None:
    logits = torch.randn(1, 100)
    assert kl_from_clean(logits, logits.clone()) == pytest.approx(0.0, abs=1e-5)


def test_kl_is_positive_and_grows_with_divergence() -> None:
    base = torch.zeros(1, 50)
    near = base.clone()
    near[0, 0] = 1.0
    far = base.clone()
    far[0, 0] = 8.0
    assert 0 < kl_from_clean(base, near) < kl_from_clean(base, far)


def test_kl_survives_fp16_inputs() -> None:
    """Model logits arrive in fp16; the computation must upcast."""
    a = torch.randn(1, 1000).half()
    b = a.clone()
    b[0, 0] += 3.0
    value = kl_from_clean(a, b)
    assert math.isfinite(value) and value > 0


def test_accuracy_of_all_correct_is_one() -> None:
    assert accuracy([True] * 10).value == pytest.approx(1.0)
    assert accuracy([False] * 10).value == pytest.approx(0.0)


# -- grading ------------------------------------------------------------------


def test_choice_result_reports_argmax_and_probabilities() -> None:
    r = ChoiceResult(options=["a", "b", "c"], logprobs=[-1.0, -0.1, -3.0])
    assert r.prediction == "b"
    assert r.argmax == 1
    assert sum(r.probs) == pytest.approx(1.0)
    assert r.prob_of("b") > r.prob_of("a") > r.prob_of("c")
    assert r.margin == pytest.approx(0.9)


def test_free_form_grading_accepts_synonyms_and_rejects_others() -> None:
    assert grade_free_form("sea", "ocean")
    assert grade_free_form("The waves.", "ocean")
    assert grade_free_form("OCEAN", "ocean")
    assert not grade_free_form("volcano", "ocean")
    assert not grade_free_form("NONE", "ocean")


def test_free_form_grading_handles_unknown_targets() -> None:
    assert grade_free_form("penguin", "penguin")
    assert not grade_free_form("walrus", "penguin")
