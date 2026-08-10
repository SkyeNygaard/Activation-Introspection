from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from introspect.codebook_icl import LABELS, VISIBLE_SAMPLES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_attention_head_screen as screen
import run_attention_localization as stage1


def test_complementary_cross_dev_design_protocol_and_compute_budget(tmp_path: Path) -> None:
    episodes = screen.complementary_dev_episodes(VISIBLE_SAMPLES[1])
    stage1_episodes = stage1.balanced_dev_episodes(VISIBLE_SAMPLES[1])

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
    assert {episode.cell_id for episode in episodes}.isdisjoint(
        episode.cell_id for episode in stage1_episodes
    )
    for episode in episodes:
        stage1_match = next(
            other
            for other in stage1_episodes
            if other.demo_signs == episode.demo_signs and other.query_sign == episode.query_sign
        )
        assert episode.positive_label != stage1_match.positive_label

    assert screen.CONCEPTS == ("bread", "volcano", "violin")
    assert screen.CARRIERS == tuple(VISIBLE_SAMPLES[1:3])
    assert screen.PAIRS == (
        (21, "query_marker"),
        (23, "query_marker"),
        (26, "final_answer"),
        (31, "final_answer"),
    )
    assert screen.expected_scored_forwards(False) == 5112
    assert screen.expected_scored_forwards(True) == 12
    smoke_episodes = screen._run_axes(True)[-1]
    assert len(smoke_episodes) == 2
    assert smoke_episodes[0].demo_signs == smoke_episodes[1].demo_signs
    assert smoke_episodes[0].positive_label == smoke_episodes[1].positive_label
    assert {episode.query_sign for episode in smoke_episodes} == {-1, 1}
    assert [episode.cell_id for episode in smoke_episodes] == [
        episode.cell_id for episode in screen._episodes(VISIBLE_SAMPLES[1], True)
    ]

    source_hashes = screen._source_hashes(Path(__file__).resolve().parents[1])
    protocol = screen.build_protocol(source_hashes, frozen_on="2026-08-10")
    assert protocol["design"]["units"] == 72
    assert protocol["design"]["arms_per_unit"] == 71
    assert protocol["design"]["candidate_components"] == 64
    assert protocol["design"]["scored_forwards"] == 5112
    assert "unweighted mean over the six" in protocol["analysis_rules"]["aggregation"]
    assert (
        "aggregate denominator must be positive" in protocol["analysis_rules"]["removal_fraction"]
    )
    assert protocol["analysis_rules"]["selection"].endswith("no top-k truncation")
    assert protocol["analysis_rules"]["sparse_go"].startswith("proceed only if 2-4")
    assert protocol["smoke_disclosure"].startswith("Smoke may use one query-twin pair")

    protocol_path = tmp_path / "protocol.json"
    raw = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path.write_text(raw)
    loaded, digest = screen.load_protocol(protocol_path, source_hashes)
    assert loaded == protocol
    assert digest == hashlib.sha256(raw.encode()).hexdigest()

    tampered = copy.deepcopy(protocol)
    tampered["design"]["selected_layer_roles"][0]["layer"] = 22
    protocol_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="does not match"):
        screen.load_protocol(protocol_path, source_hashes)

    destination = tmp_path / "artifact.json"
    first = tmp_path / ".first.tmp"
    first.write_text("complete\n")
    screen._publish_no_overwrite(first, destination)
    assert destination.read_text() == "complete\n"
    second = tmp_path / ".second.tmp"
    second.write_text("must not overwrite\n")
    with pytest.raises(FileExistsError):
        screen._publish_no_overwrite(second, destination)
    assert destination.read_text() == "complete\n"
