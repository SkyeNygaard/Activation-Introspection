"""Guards on concept-vector construction.

These encode failures that were found the hard way on Qwen2.5-0.5B and that are
silent: nothing errors, the numbers look plausible, and every downstream
identification result is meaningless.
"""

from __future__ import annotations

import pytest
import torch

from introspect.concepts import (
    MIN_CONTEXT_TOKENS,
    TEMPLATES,
    ConceptVector,
    SinkPositionError,
    _last_token,
    center_bank,
    max_offdiagonal_cosine,
    random_control,
    shuffled_control,
)
from introspect.models import LoadedModel


class _Tok:
    """Tokenizer stub: one token per whitespace-separated word."""

    def __call__(self, text: str, return_tensors: str = "pt") -> object:
        n = max(len(text.split()), 1)
        return type("E", (), {"input_ids": torch.zeros(1, n, dtype=torch.long)})()


def _lm() -> LoadedModel:
    return LoadedModel(
        name="stub",
        model=None,  # type: ignore[arg-type]
        tokenizer=_Tok(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_short_prompt_is_rejected_before_it_reads_the_sink() -> None:
    """A 1-token prompt captures at position 0, where the massive activation lives.

    On Qwen2.5-0.5B that coordinate hits ~1537 against a typical residual norm of
    ~20, so a single short prompt in the averaging set drives every concept vector
    to cosine 1.00 with every other.
    """
    with pytest.raises(SinkPositionError, match="attention sink"):
        _last_token(_lm(), "thing", layer=0)


def test_all_templates_clear_the_minimum_context() -> None:
    lm = _lm()
    for template in TEMPLATES:
        for filler in ("thing", "item", "object"):
            text = template.format(concept=filler)
            n = int(lm.encode(text).shape[1])
            assert n >= MIN_CONTEXT_TOKENS, f"{text!r} is only {n} tokens"


def test_all_templates_end_on_the_concept() -> None:
    """Trailing punctuation moves the capture off the concept token."""
    for template in TEMPLATES:
        assert template.rstrip().endswith("{concept}"), template


def test_centering_removes_a_shared_component() -> None:
    shared = torch.zeros(8)
    shared[0] = 50.0
    bank = {}
    for i in range(4):
        distinct = torch.zeros(8)
        distinct[i + 1] = 1.0
        bank[f"c{i}"] = ConceptVector(name=f"c{i}", layer=0, vector=shared + distinct)

    assert max_offdiagonal_cosine(bank) > 0.99
    assert max_offdiagonal_cosine(center_bank(bank)) < 0.5


def test_controls_match_norm_but_not_direction() -> None:
    ref = ConceptVector(name="c", layer=0, vector=torch.randn(64) * 3)
    for control in (random_control(ref, seed=1), shuffled_control(ref, seed=1)):
        assert torch.allclose(control.vector.norm(), ref.vector.norm(), rtol=1e-4)
        assert abs(float(torch.dot(control.unit(), ref.unit()))) < 0.5


def test_shuffled_control_preserves_the_coordinate_multiset() -> None:
    ref = ConceptVector(name="c", layer=0, vector=torch.randn(64))
    shuffled = shuffled_control(ref, seed=3)
    assert torch.allclose(ref.vector.sort().values, shuffled.vector.sort().values)
