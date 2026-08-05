"""Explicitly labeled descriptive plots for scripted mechanism checks.

Legacy figure entry points that implied causal feedback, confidence intervals,
validated defense performance, or uncensored leak-time analysis now fail loudly.
Existing PNG artifacts remain on disk for provenance and are labeled legacy in
``figures/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sandbox.runner import Condition

BLUE = "#1b6ca8"
GREY = "#9a9a9a"


def plot_scripted_feedback(conditions: list[Condition], path: Path) -> None:
    """Plot authored-policy outcomes without inferential error bars."""
    order = ["silent", "generic", "specific"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    for agent, colour, marker in (("adaptive", BLUE, "o"), ("static", GREY, "s")):
        xs, values = [], []
        for index, feedback in enumerate(order):
            matches = [
                condition
                for condition in conditions
                if condition.feedback == feedback and condition.agent == agent
            ]
            if matches:
                xs.append(index)
                values.append(matches[0].summary().no_leak_fraction.require_mean())
        ax.plot(xs, values, marker=marker, label=f"scripted {agent} policy", color=colour)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["silent", "generic", "specific"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("feedback string passed to the authored response rule")
    ax.set_ylabel("fraction of scheduler variants with no labeled leak")
    ax.set_title(
        "SCRIPTED MECHANISM DIAGNOSTIC\n"
        "response to feedback is encoded by construction; descriptive means only",
        fontsize=10,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scripted_tradeoff(rows: list[tuple[str, float, float]], path: Path) -> None:
    """Plot sandbox no-leak/task-completion means for authored monitors."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    seen: dict[tuple[float, float], int] = {}
    for label, no_leak, task_completion in rows:
        ax.plot(task_completion, no_leak, marker="o", color=BLUE)
        key = (round(task_completion, 2), round(no_leak, 2))
        rank = seen.get(key, 0)
        seen[key] = rank + 1
        ax.annotate(
            label,
            (task_completion, no_leak),
            textcoords="offset points",
            xytext=(-64 if task_completion > 0.9 else 6, 5 + rank * 11),
            fontsize=8,
        )
    ax.set_xlabel("mean honest task completion in sandbox schedules")
    ax.set_ylabel("mean no-leak indicator for scripted attacker")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.set_title(
        "SCRIPTED SANDBOX TRADE-OFF DIAGNOSTIC\n"
        "not a validated defense frontier; descriptive means only",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feedback(conditions: list[Condition], path: Path) -> None:
    """Retired legacy entry point; use ``plot_scripted_feedback``."""
    del conditions, path
    raise RuntimeError("legacy causal-feedback figure retired; use plot_scripted_feedback")


def plot_frontier(rows: list[tuple[str, float, float, float, float]], path: Path) -> None:
    """Retired legacy entry point; use ``plot_scripted_tradeoff``."""
    del rows, path
    raise RuntimeError("legacy validated-defense frontier retired; use plot_scripted_tradeoff")


def plot_leak_delay(rows: list[tuple[str, float, float, float]], path: Path) -> None:
    """Retired because legacy means imputed right-censored runs as event times."""
    del rows, path
    raise RuntimeError("legacy leak-delay figure retired; use a registered survival analysis")


def plot_llm_adaptation(json_path: Path, path: Path) -> None:
    """Retired because deterministic menu orders cannot supply intervals."""
    del json_path, path
    raise RuntimeError(
        "legacy LLM adaptation figure retired: exact menu orders are nuisance "
        "conditions and cannot supply confidence intervals"
    )
