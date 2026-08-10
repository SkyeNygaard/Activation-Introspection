from __future__ import annotations

import pytest
import torch

from introspect.codebook_icl import (
    LABELS,
    VISIBLE_SAMPLES,
    Episode,
    condition_interventions,
    exact_episodes,
)
from introspect.concepts import ConceptVector


def test_exact_design_balances_every_hidden_and_visible_nuisance() -> None:
    episodes = exact_episodes(VISIBLE_SAMPLES[0])

    assert len(episodes) == 24
    assert len({episode.cell_id for episode in episodes}) == 24
    assert len({episode.demo_signs for episode in episodes}) == 6
    assert {episode.query_sign for episode in episodes} == {-1, 1}
    assert {(e.positive_label, e.negative_label) for e in episodes} == {
        ("Q", "K"),
        ("K", "Q"),
    }
    assert [sum(e.correct_label == label for e in episodes) for label in LABELS] == [12, 12]
    for episode in episodes:
        prompt = episode.render_user()
        assert episode.demo_signs.count(1) == episode.demo_signs.count(-1) == 2
        assert prompt.count(f"Observation: {VISIBLE_SAMPLES[0]}") == 5
        assert prompt.count("Label: Q") == prompt.count("Label: K") == 2


def test_episode_rejects_unbalanced_demos() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        Episode("bad", (1, 1, 1, -1), 1, "Q", "K", VISIBLE_SAMPLES[0])


def test_test_only_withholds_demonstration_interventions() -> None:
    vector = ConceptVector(name="target", layer=3, vector=torch.arange(1.0, 5.0))
    positions = (4, 8, 12, 16, 20)
    signs = (-1, 1, -1, 1, 1)

    target = condition_interventions("target", vector, positions, signs, strength=0.25)
    test_only = condition_interventions("test_only", vector, positions, signs, strength=0.25)

    assert sorted(p for iv in target for p in iv.positions) == sorted(positions)
    assert [p for iv in test_only for p in iv.positions] == [positions[-1]]
    assert torch.equal(test_only[0].direction, vector.vector)
    assert all(iv.per_position for iv in [*target, *test_only])
    assert condition_interventions("clean", None, positions, signs, strength=0.25) == []
    with pytest.raises(ValueError, match="needs a direction"):
        condition_interventions("random", None, positions, signs, strength=0.25)
