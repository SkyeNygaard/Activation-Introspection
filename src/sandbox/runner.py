"""Scripted sandbox mechanism diagnostics across scheduler variants.

This module does not perform statistical inference. The attacker response rule,
task bank, monitor, and staging-path distribution are all researcher-authored.
Changing ``scheduler_variants`` changes task schedules and staging filenames; it
does not sample independent language-model behavior or a deployment population.

Summaries therefore contain means and observed ranges only. First-leak times keep
right-censored runs as ``None`` and report the censor count explicitly. The
module intentionally exposes no significance verdict, confidence interval, or
pooled detection metric.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sandbox.actions import FeedbackLevel
from sandbox.agents import AdaptiveAttacker, AgentProtocol, HonestAgent, StaticAttacker
from sandbox.env import RunResult, run
from sandbox.metrics import mean_suspicion_curve
from sandbox.monitors import (
    CrossEpisodeMonitor,
    Monitor,
    NullMonitor,
    RuleMonitor,
    Visibility,
    WindowedMonitor,
)

AgentFactory = Callable[[int], AgentProtocol]
MonitorFactory = Callable[[], Monitor]

AGENTS: dict[str, AgentFactory] = {
    "honest": lambda variant: HonestAgent(),
    "static": lambda variant: StaticAttacker(seed=variant),
    "adaptive": lambda variant: AdaptiveAttacker(seed=variant),
}

MONITORS: dict[str, MonitorFactory] = {
    "none": NullMonitor,
    "actions": lambda: RuleMonitor(Visibility.ACTIONS_ONLY),
    "actions+cot": lambda: RuleMonitor(Visibility.ACTIONS_AND_COT),
    "actions+summary": lambda: RuleMonitor(Visibility.ACTIONS_AND_SUMMARY),
    "window1": lambda: WindowedMonitor(1),
    "window2": lambda: WindowedMonitor(2),
    "window4": lambda: WindowedMonitor(4),
    "cross-episode": CrossEpisodeMonitor,
}


@dataclass(frozen=True)
class DescriptiveSummary:
    """Observed mean and range, with no inferential interpretation."""

    mean: float | None
    minimum: float | None
    maximum: float | None
    n: int

    def __str__(self) -> str:
        if self.mean is None or self.minimum is None or self.maximum is None:
            return "no observed values"
        return f"mean={self.mean:.2f}; range=[{self.minimum:.2f}, {self.maximum:.2f}]; n={self.n}"

    def require_mean(self) -> float:
        if self.mean is None:
            raise ValueError("no observed mean")
        return self.mean

    def require_range(self) -> tuple[float, float]:
        if self.minimum is None or self.maximum is None:
            raise ValueError("no observed range")
        return self.minimum, self.maximum


def describe(values: list[float]) -> DescriptiveSummary:
    if not values:
        return DescriptiveSummary(None, None, None, 0)
    arr = np.array(values, dtype=float)
    return DescriptiveSummary(
        mean=float(arr.mean()),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        n=len(arr),
    )


def paired_descriptive_difference(first: list[float], second: list[float]) -> DescriptiveSummary:
    """Describe paired scheduler-variant differences without a hypothesis test."""
    if len(first) != len(second):
        raise ValueError(f"paired diagnostics must match, got {len(first)} and {len(second)}")
    return describe([x - y for x, y in zip(first, second, strict=True)])


@dataclass(frozen=True)
class ConditionSummary:
    no_leak_fraction: DescriptiveSummary
    task_completion_fraction: DescriptiveSummary
    blocked_actions: DescriptiveSummary
    observed_first_leak_episode: DescriptiveSummary
    right_censored_runs: int
    total_runs: int
    suspicion_slope_diagnostic: DescriptiveSummary


@dataclass
class Condition:
    """One scripted configuration evaluated across scheduler/path variants."""

    agent: str
    monitor: str
    feedback: str
    n_episodes: int
    results: list[RunResult] = field(default_factory=list)

    @property
    def no_leak(self) -> list[float]:
        return [0.0 if result.secret_leaked else 1.0 for result in self.results]

    @property
    def task_completion(self) -> list[float]:
        return [result.usefulness for result in self.results]

    @property
    def blocked_actions(self) -> list[float]:
        return [
            float(sum(episode.blocked_count for episode in result.episodes))
            for result in self.results
        ]

    @property
    def first_leak_episode(self) -> list[int | None]:
        """Observed event time or ``None`` when right-censored at the horizon."""
        return [result.leaked_at_episode for result in self.results]

    @property
    def observed_first_leak_episode(self) -> list[float]:
        return [float(episode) for episode in self.first_leak_episode if episode is not None]

    @property
    def right_censored_runs(self) -> int:
        return sum(episode is None for episode in self.first_leak_episode)

    @property
    def suspicion_slope_diagnostic(self) -> list[float]:
        """Descriptive trend in monitor scores; not an obfuscation estimand."""
        slopes = []
        for result in self.results:
            curve = mean_suspicion_curve(result)
            if len(curve) < 2:
                continue
            x = np.arange(len(curve), dtype=float)
            slopes.append(float(np.polyfit(x, np.array(curve), 1)[0]))
        return slopes

    def summary(self) -> ConditionSummary:
        return ConditionSummary(
            no_leak_fraction=describe(self.no_leak),
            task_completion_fraction=describe(self.task_completion),
            blocked_actions=describe(self.blocked_actions),
            observed_first_leak_episode=describe(self.observed_first_leak_episode),
            right_censored_runs=self.right_censored_runs,
            total_runs=len(self.results),
            suspicion_slope_diagnostic=describe(self.suspicion_slope_diagnostic),
        )


def run_condition(
    agent: str,
    monitor: str,
    *,
    feedback: FeedbackLevel = FeedbackLevel.SPECIFIC,
    n_episodes: int = 10,
    scheduler_variants: int = 20,
) -> Condition:
    """Execute authored mechanism checks over task/path scheduler variants."""
    condition = Condition(
        agent=agent,
        monitor=monitor,
        feedback=feedback.value,
        n_episodes=n_episodes,
    )
    for variant in range(scheduler_variants):
        condition.results.append(
            run(
                AGENTS[agent](variant),
                MONITORS[monitor](),
                label=(
                    f"scripted diagnostic: {agent} vs {monitor} "
                    f"[{feedback.value}] scheduler_variant={variant}"
                ),
                n_episodes=n_episodes,
                feedback=feedback,
                seed=variant,
            )
        )
    return condition


def save(conditions: list[Condition], path: Path) -> None:
    """Write machine-readable descriptive diagnostics, including censor flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for condition in conditions:
            handle.write(
                json.dumps(
                    {
                        "artifact_kind": "scripted_mechanism_diagnostic",
                        "claim_scope": (
                            "descriptive outcomes of a researcher-authored sandbox policy; "
                            "not model behavior, causal inference, or a validated detector"
                        ),
                        "condition": {
                            "agent": condition.agent,
                            "monitor": condition.monitor,
                            "feedback": condition.feedback,
                            "n_episodes": condition.n_episodes,
                        },
                        "summary": asdict(condition.summary()),
                        "runs": [
                            {
                                "scheduler_variant": result.seed,
                                "no_leak": not result.secret_leaked,
                                "task_completion_fraction": result.usefulness,
                                "first_leak_episode": result.leaked_at_episode,
                                "right_censored": result.leaked_at_episode is None,
                                "blocked_actions": sum(
                                    episode.blocked_count for episode in result.episodes
                                ),
                            }
                            for result in condition.results
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
