"""Checks for the memory/concurrency guard.

The live machine state changes between runs, so these pin the parsing rules
rather than any particular measurement. The regex case below is the real one
that slipped through: a sibling session launched `python run_dev.py` with no
leading slash and the first version of the pattern missed it, so a 3B job
started alongside a 4B job and was killed by the OS mid-load.
"""

from __future__ import annotations

import pytest

from introspect.preflight import _RUNNER_RE, available_gib, required_gib


@pytest.mark.parametrize(
    "command",
    [
        "/Users/x/proj/.venv/bin/python run_dev.py --model Qwen/Qwen3-4B-Instruct-2507",
        "/Users/x/proj/.venv/bin/python /Users/x/proj/scripts/run_remap_training.py --seed 0",
        "python run_report_training.py",
    ],
)
def test_detects_a_runner_however_it_was_invoked(command: str) -> None:
    match = _RUNNER_RE.search(command)
    assert match is not None
    assert match.group(1).startswith("run_")


@pytest.mark.parametrize(
    "command",
    [
        "/Users/x/proj/.venv/bin/python -m introspect.preflight qwen-3b",
        "/Users/x/proj/.venv/bin/python -m pytest tests/",
        "/bin/zsh -c source ~/.claude/shell-snapshots/snapshot.sh",
    ],
)
def test_ignores_non_runners(command: str) -> None:
    assert _RUNNER_RE.search(command) is None


def test_training_requires_more_than_inference() -> None:
    assert required_gib("qwen-3b", training=True) > required_gib("qwen-3b", training=False)
    assert required_gib("qwen-0.5b", training=True) < required_gib("qwen-3b", training=True)


def test_unmeasured_model_refuses_rather_than_guessing() -> None:
    with pytest.raises(SystemExit):
        required_gib("llama-70b", training=False)


def test_available_memory_is_plausible() -> None:
    """Parses vm_stat without asserting a number that varies between runs."""
    free = available_gib()
    assert 0.0 < free < 1024.0
