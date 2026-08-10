from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from introspect.codebook_icl import LABELS, VISIBLE_SAMPLES, PreparedEpisode
from introspect.models import LoadedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_attention_localization import (
    _demo_label_positions,
    balanced_dev_episodes,
    receiver_roles,
)


class _CharacterTokenizer:
    def __call__(self, text: str, **_kwargs: object) -> Any:
        return SimpleNamespace(
            input_ids=torch.tensor([[ord(character) for character in text]]),
            offset_mapping=torch.tensor([[[index, index + 1] for index in range(len(text))]]),
        )


def test_dev_subset_balances_orders_mappings_query_twins_and_answers() -> None:
    episodes = balanced_dev_episodes(VISIBLE_SAMPLES[0])

    assert len(episodes) == 12
    assert len({episode.demo_signs for episode in episodes}) == 6
    assert {episode.query_sign for episode in episodes} == {-1, 1}
    assert [sum(episode.positive_label == label for episode in episodes) for label in LABELS] == [
        6,
        6,
    ]
    assert [sum(episode.correct_label == label for episode in episodes) for label in LABELS] == [
        6,
        6,
    ]

    by_order = {
        order: [episode for episode in episodes if episode.demo_signs == order]
        for order in {episode.demo_signs for episode in episodes}
    }
    assert all(len(pair) == 2 for pair in by_order.values())
    assert all({episode.query_sign for episode in pair} == {-1, 1} for pair in by_order.values())
    assert all(pair[0].render_user() == pair[1].render_user() for pair in by_order.values())


def test_receiver_roles_use_span_specific_demo_labels_and_query_marker() -> None:
    episode = balanced_dev_episodes(VISIBLE_SAMPLES[0])[0]
    prompt = episode.render_user() + "\nLabel:"
    token_ids = torch.tensor([[ord(character) for character in prompt]])
    state_positions = tuple(index for index, character in enumerate(prompt) if character == "§")
    prepared = PreparedEpisode(
        episode=episode,
        prompt=prompt,
        input_ids=token_ids,
        state_positions=state_positions,
        label_ids=(ord("Q"), ord("K")),
    )
    model = LoadedModel(
        name="stub",
        model=nn.Identity(),
        tokenizer=_CharacterTokenizer(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    labels = _demo_label_positions(model, prepared)
    roles = receiver_roles(prepared, labels)

    assert len(labels) == 4
    assert roles["demo_labels"] == labels
    assert roles["query_marker"] == (state_positions[-1],)
    assert roles["final_answer"] == (len(prompt) - 1,)
    assert roles["all_positions"] == "all"
