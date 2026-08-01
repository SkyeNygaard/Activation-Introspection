"""The three experiments from notes/02-experiment-plan.md, with intervals.

1. Feedback sweep -- does obfuscation scale with how much the refusal leaks?
2. Monitor-memory sweep -- where is the knee between per-episode and full history?
3. Detection at a staffable audit budget.

All scripted, no API key, deterministic given the seed list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sandbox.actions import FeedbackLevel
from sandbox.runner import Condition, detection_metrics, difference, run_condition, save

console = Console()


def experiment_feedback(seeds: int, episodes: int) -> list[Condition]:
    """Does the attacker learn to look boring, and does that scale with feedback?

    Prediction (recorded before running): obfuscation slope is monotone in
    feedback specificity and near zero under SILENT. The static attacker is the
    control -- its slope should be flat under every feedback level, because it
    never adapts.
    """
    console.rule("1. feedback sweep")
    conds = []
    table = Table("feedback", "agent", "safety", "useful", "leak ep", "obfusc. slope")
    for fb in (FeedbackLevel.SILENT, FeedbackLevel.GENERIC, FeedbackLevel.SPECIFIC):
        for agent in ("static", "adaptive"):
            c = run_condition(agent, "actions+cot", feedback=fb, n_episodes=episodes, seeds=seeds)
            conds.append(c)
            s = c.summary()
            table.add_row(
                fb.value,
                agent,
                str(s["safety"]),
                str(s["usefulness"]),
                str(s["leak_episode"]),
                str(s["obfuscation_slope"]),
            )
        table.add_section()
    console.print(table)

    console.print("\n[bold]adaptive minus static, obfuscation slope (paired over seeds):[/bold]")
    for fb in (FeedbackLevel.SILENT, FeedbackLevel.GENERIC, FeedbackLevel.SPECIFIC):
        adaptive = next(c for c in conds if c.feedback == fb.value and c.agent == "adaptive")
        static = next(c for c in conds if c.feedback == fb.value and c.agent == "static")
        d = difference(adaptive.obfuscation_slope, static.obfuscation_slope)
        verdict = "significant" if d.excludes_zero else "not distinguishable from zero"
        console.print(f"  {fb.value:<9} {d}  [dim]{verdict}[/dim]")
    return conds


def experiment_memory(seeds: int, episodes: int) -> list[Condition]:
    """How much monitor memory is needed, and what does it cost?"""
    console.rule("2. monitor memory sweep")
    monitors = ["none", "actions", "actions+cot", "window1", "window2", "window4", "cross-episode"]
    conds = []
    table = Table(
        "monitor",
        "safety (adaptive)",
        "leak episode",
        "usefulness (honest)",
        "honest blocked",
        "AUROC",
        "catch@5%FPR",
    )

    benign = {m: run_condition("honest", m, n_episodes=episodes, seeds=seeds) for m in monitors}
    for m in monitors:
        attack = run_condition("adaptive", m, n_episodes=episodes, seeds=seeds)
        conds.extend([attack, benign[m]])
        au, catch = detection_metrics(attack, benign[m])
        table.add_row(
            m,
            str(attack.summary()["safety"]),
            str(attack.summary()["leak_episode"]),
            str(benign[m].summary()["usefulness"]),
            str(benign[m].summary()["blocked"]),
            f"{au:.2f}",
            f"{catch:.2f}",
        )
    console.print(table)
    console.print(
        "\n[dim]Safety is measured against the adaptive attacker; usefulness and blocked "
        "counts against the honest agent, which is the only fair reading of the "
        "false-positive cost. Leak episode is censored at the run length, so a monitor "
        "that merely delays the attack is visible here even when safety reads 0.00.[/dim]"
    )
    return conds


def experiment_adaptation(seeds: int, episodes: int) -> list[Condition]:
    """Does adaptation matter, holding everything else fixed?"""
    console.rule("3. adaptation effect")
    adaptive = run_condition("adaptive", "actions+cot", n_episodes=episodes, seeds=seeds)
    static = run_condition("static", "actions+cot", n_episodes=episodes, seeds=seeds)

    d_safety = difference(static.safety, adaptive.safety)
    console.print(f"  safety, static minus adaptive: {d_safety}")
    console.print(
        "  [dim]positive means adaptation defeats the monitor; the two attackers run "
        "identical code except for whether blocks escalate the strategy[/dim]"
    )
    return [adaptive, static]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/sweep.jsonl"))
    ap.add_argument("--figures", type=Path, default=None, help="write plots to this directory")
    args = ap.parse_args()

    conds = []
    fb = experiment_feedback(args.seeds, args.episodes)
    mem = experiment_memory(args.seeds, args.episodes)
    conds += fb + mem + experiment_adaptation(args.seeds, args.episodes)

    if args.figures is not None:
        from sandbox.figures import plot_feedback, plot_frontier, plot_leak_delay

        args.figures.mkdir(parents=True, exist_ok=True)
        plot_feedback(fb, args.figures / "feedback.png")

        monitors = [
            "none",
            "actions",
            "actions+cot",
            "window1",
            "window2",
            "window4",
            "cross-episode",
        ]
        frontier, delay = [], []
        for m in monitors:
            attack = next(c for c in mem if c.monitor == m and c.agent == "adaptive")
            benign = next(c for c in mem if c.monitor == m and c.agent == "honest")
            a, b = attack.summary(), benign.summary()
            frontier.append(
                (
                    m,
                    a["safety"].value,
                    (a["safety"].hi - a["safety"].lo) / 2,
                    b["usefulness"].value,
                    (b["usefulness"].hi - b["usefulness"].lo) / 2,
                )
            )
            le = a["leak_episode"]
            delay.append((m, le.value, le.lo, le.hi))
        plot_frontier(frontier, args.figures / "frontier.png")
        plot_leak_delay(delay, args.figures / "leak_delay.png")
        console.print(f"wrote 3 figures to {args.figures}")

    save(conds, args.out)
    console.print(f"\nwrote {len(conds)} conditions to {args.out}")


if __name__ == "__main__":
    main()
