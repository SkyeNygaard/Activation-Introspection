"""Checks for the memory/concurrency guard.

The live machine state changes between runs, so these pin the parsing rules
rather than any particular measurement. The regex case below is the real one
that slipped through: a sibling session launched `python run_dev.py` with no
leading slash and the first version of the pattern missed it, so a 3B job
started alongside a 4B job and was killed by the OS mid-load.
"""

from __future__ import annotations

import pytest

from introspect.preflight import _RUNNER_RE, _competing_runs, available_gib, check, required_gib


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


def test_competing_runs_excludes_uv_parent_but_keeps_sibling() -> None:
    output = "\n".join(
        [
            "10 1 uv run python scripts/run_natural_state.py --smoke",
            "20 10 python scripts/run_natural_state.py --smoke",
            "30 1 python scripts/run_report_training.py --seed 0",
        ]
    )

    assert _competing_runs(output, mine=20) == [(30, "run_report_training.py")]


def test_remote_provider_runs_are_not_gpu_competitors() -> None:
    """A Codex-provider run sends its work to an API and holds no GPU memory.

    Both are runners by name. Only the local one competes for MPS.
    """
    output = "\n".join(
        [
            "10 1 python run_scaled.py --provider codex --model gpt-5.6-luna --workers 4",
            "20 1 python scripts/run_report_training.py --seed 0",
        ]
    )

    assert _competing_runs(output, mine=99) == [(20, "run_report_training.py")]


def test_runner_without_a_provider_flag_is_still_a_competitor() -> None:
    """The clearance is positive: silence about the provider means local."""
    output = "10 1 python run_scaled.py --model qwen-3b --workers 4"

    assert _competing_runs(output, mine=99) == [(10, "run_scaled.py")]


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


def test_slack_allows_a_declared_shortfall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in slack lets a run start short, and only by the amount declared."""
    monkeypatch.setattr("introspect.preflight.available_gib", lambda: 10.0)
    monkeypatch.setattr("introspect.preflight.competing_runs", lambda: [])
    need = required_gib("qwen-3b", training=True)  # 12.4

    monkeypatch.delenv("INTROSPECT_PREFLIGHT_SLACK_GIB", raising=False)
    with pytest.raises(SystemExit):
        check("qwen-3b", training=True)

    monkeypatch.setenv("INTROSPECT_PREFLIGHT_SLACK_GIB", "4")
    check("qwen-3b", training=True)  # 10.0 >= 12.4 - 4

    monkeypatch.setenv("INTROSPECT_PREFLIGHT_SLACK_GIB", "1")
    with pytest.raises(SystemExit):
        check("qwen-3b", training=True)
    assert need > 10.0
