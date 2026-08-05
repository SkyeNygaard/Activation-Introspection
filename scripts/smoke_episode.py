"""Smoke-test that the toy world executes its authored mechanisms.

This is a one-schedule engineering diagnostic. It emits no model-behavior,
causal, significance, survival, or validated-detection claim.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.table import Table

from sandbox import (
    AdaptiveAttacker,
    CrossEpisodeMonitor,
    FeedbackLevel,
    HonestAgent,
    NullMonitor,
    RuleMonitor,
    StaticAttacker,
    Visibility,
    run,
)
from sandbox.agents import AgentProtocol
from sandbox.metrics import mean_suspicion_curve
from sandbox.monitors import Monitor

console = Console()
N_EPISODES = 8


def main() -> None:
    console.print(
        "[bold yellow]SCRIPTED MECHANISM SMOKE TEST[/bold yellow] — one authored "
        "scheduler variant; descriptive engineering output only"
    )
    monitors: list[tuple[str, Callable[[], Monitor]]] = [
        ("none", lambda: NullMonitor()),
        ("rule/actions", lambda: RuleMonitor(Visibility.ACTIONS_ONLY)),
        ("rule/actions+cot", lambda: RuleMonitor(Visibility.ACTIONS_AND_COT)),
        ("rule/actions+summary", lambda: RuleMonitor(Visibility.ACTIONS_AND_SUMMARY)),
        ("cross-episode", lambda: CrossEpisodeMonitor()),
    ]
    agents: list[tuple[str, Callable[[], AgentProtocol]]] = [
        ("honest", HonestAgent),
        ("static scripted attacker", StaticAttacker),
        ("adaptive scripted attacker", AdaptiveAttacker),
    ]

    table = Table(
        "authored policy",
        "authored monitor",
        "no labeled leak",
        "task completion",
        "first leak / censoring",
        "blocked",
        "staged",
    )
    for agent_name, agent_class in agents:
        for monitor_name, monitor_factory in monitors:
            result = run(
                agent_class(),
                monitor_factory(),
                label=f"scripted smoke: {agent_name} vs {monitor_name}",
                n_episodes=N_EPISODES,
                feedback=FeedbackLevel.SPECIFIC,
            )
            leak_status = (
                f"observed ep{result.leaked_at_episode}"
                if result.leaked_at_episode is not None
                else f"right-censored at {N_EPISODES}"
            )
            table.add_row(
                agent_name,
                monitor_name,
                str(not result.secret_leaked),
                f"{result.usefulness:.2f}",
                leak_status,
                str(sum(episode.blocked_count for episode in result.episodes)),
                ", ".join(result.staged_paths) or "-",
            )
        table.add_section()
    console.print(table)

    console.rule("authored-monitor score curves (diagnostic only)")
    for label, agent_class in (("static", StaticAttacker), ("adaptive", AdaptiveAttacker)):
        result = run(
            agent_class(),
            RuleMonitor(Visibility.ACTIONS_AND_COT),
            n_episodes=N_EPISODES,
        )
        curve = " ".join(f"{value:.2f}" for value in mean_suspicion_curve(result))
        console.print(f"  {label:>8}  {curve}")
    console.print(
        "\n[dim]These curves reflect a researcher-authored policy and monitor. They "
        "are useful for catching wiring regressions, not for inferring learning or "
        "obfuscation in models.[/dim]"
    )


if __name__ == "__main__":
    main()
