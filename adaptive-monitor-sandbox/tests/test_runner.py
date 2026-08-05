"""Regression tests for the non-inferential scripted diagnostic runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from sandbox.figures import plot_feedback, plot_frontier, plot_leak_delay
from sandbox.runner import paired_descriptive_difference, run_condition, save


def test_scripted_summary_keeps_right_censoring_explicit() -> None:
    condition = run_condition(
        "static",
        "actions+cot",
        n_episodes=3,
        scheduler_variants=4,
    )
    summary = condition.summary()

    assert condition.first_leak_episode == [None, None, None, None]
    assert summary.right_censored_runs == 4
    assert summary.observed_first_leak_episode.n == 0


def test_saved_scripted_output_has_no_inferential_or_detection_fields(tmp_path: Path) -> None:
    condition = run_condition(
        "static",
        "actions+cot",
        n_episodes=3,
        scheduler_variants=2,
    )
    output = tmp_path / "diagnostic.jsonl"
    save([condition], output)
    record = json.loads(output.read_text())

    assert record["artifact_kind"] == "scripted_mechanism_diagnostic"
    assert record["summary"]["right_censored_runs"] == 2
    assert record["summary"]["observed_first_leak_episode"]["mean"] is None
    assert all(run["right_censored"] for run in record["runs"])
    assert all(run["first_leak_episode"] is None for run in record["runs"])
    serialized = json.dumps(record).lower()
    for forbidden in ("confidence_interval", "p_value", "significant", "auroc", "catch_rate"):
        assert forbidden not in serialized


def test_paired_difference_is_descriptive_only() -> None:
    summary = paired_descriptive_difference([1.0, 0.0], [0.0, 0.0])
    assert summary.mean is not None
    assert summary.mean == pytest.approx(0.5)
    assert not hasattr(summary, "excludes_zero")


@pytest.mark.parametrize(
    "legacy_plot,args",
    [
        (plot_feedback, ([],)),
        (plot_frontier, ([],)),
        (plot_leak_delay, ([],)),
    ],
)
def test_legacy_misleading_plot_entry_points_fail_loudly(
    legacy_plot: Callable[..., None], args: tuple[list[object]]
) -> None:
    with pytest.raises(RuntimeError):
        legacy_plot(*args, Path("unused.png"))
