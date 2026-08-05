"""Run descriptive mechanism checks for the researcher-authored sandbox policy.

These diagnostics answer whether the implementation expresses its scripted
mechanisms under different task schedules and staging paths. They do not test a
language model, estimate a causal feedback effect, validate a detector, or
perform statistical inference. First-leak times retain censoring explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sandbox.actions import FeedbackLevel
from sandbox.runner import (
    Condition,
    ConditionSummary,
    paired_descriptive_difference,
    run_condition,
    save,
)

console = Console()


def leak_observation(summary: ConditionSummary, horizon: int) -> str:
    observed = summary.total_runs - summary.right_censored_runs
    if observed == 0:
        return f"0/{summary.total_runs} observed; all censored at {horizon}"
    conditional = summary.observed_first_leak_episode
    minimum, maximum = conditional.require_range()
    return (
        f"{observed}/{summary.total_runs} observed; observed-only "
        f"mean={conditional.require_mean():.2f}, range=[{minimum:.0f}, {maximum:.0f}]; "
        f"censored={summary.right_censored_runs}"
    )


def feedback_mechanism_diagnostic(scheduler_variants: int, episodes: int) -> list[Condition]:
    console.rule("scripted feedback-response mechanism diagnostic")
    conditions = []
    table = Table(
        "feedback text",
        "scripted policy",
        "no-leak fraction",
        "task completion",
        "first-leak observation",
        "score-slope diagnostic",
    )
    for feedback in FeedbackLevel:
        for agent in ("static", "adaptive"):
            condition = run_condition(
                agent,
                "actions+cot",
                feedback=feedback,
                n_episodes=episodes,
                scheduler_variants=scheduler_variants,
            )
            conditions.append(condition)
            summary = condition.summary()
            table.add_row(
                feedback.value,
                agent,
                str(summary.no_leak_fraction),
                str(summary.task_completion_fraction),
                leak_observation(summary, episodes),
                str(summary.suspicion_slope_diagnostic),
            )
        table.add_section()
    console.print(table)

    console.print(
        "\n[dim]Paired score-slope differences below are descriptive across authored "
        "scheduler/path variants. No p-value, confidence interval, or significance "
        "verdict is defined.[/dim]"
    )
    for feedback in FeedbackLevel:
        adaptive = next(
            c for c in conditions if c.feedback == feedback.value and c.agent == "adaptive"
        )
        static = next(c for c in conditions if c.feedback == feedback.value and c.agent == "static")
        delta = paired_descriptive_difference(
            adaptive.suspicion_slope_diagnostic,
            static.suspicion_slope_diagnostic,
        )
        console.print(f"  {feedback.value:<9} adaptive - static: {delta}")
    return conditions


def monitor_mechanism_diagnostic(scheduler_variants: int, episodes: int) -> list[Condition]:
    console.rule("scripted monitor-memory mechanism diagnostic")
    monitors = ["none", "actions", "actions+cot", "window1", "window2", "window4", "cross-episode"]
    conditions = []
    table = Table(
        "authored monitor",
        "attacker no-leak fraction",
        "honest task completion",
        "honest blocked actions",
        "first-leak observation",
    )

    honest = {
        monitor: run_condition(
            "honest",
            monitor,
            n_episodes=episodes,
            scheduler_variants=scheduler_variants,
        )
        for monitor in monitors
    }
    for monitor in monitors:
        attack = run_condition(
            "adaptive",
            monitor,
            n_episodes=episodes,
            scheduler_variants=scheduler_variants,
        )
        conditions.extend([attack, honest[monitor]])
        attack_summary = attack.summary()
        honest_summary = honest[monitor].summary()
        table.add_row(
            monitor,
            str(attack_summary.no_leak_fraction),
            str(honest_summary.task_completion_fraction),
            str(honest_summary.blocked_actions),
            leak_observation(attack_summary, episodes),
        )
    console.print(table)
    console.print(
        "\n[dim]These are outcomes of authored policies in a toy world. The script "
        "does not emit AUROC/catch-rate claims because episode labels are dependent "
        "and no calibration/evaluation split exists.[/dim]"
    )
    return conditions


def hard_coded_policy_contrast(scheduler_variants: int, episodes: int) -> list[Condition]:
    console.rule("hard-coded response-policy contrast")
    adaptive = run_condition(
        "adaptive",
        "actions+cot",
        n_episodes=episodes,
        scheduler_variants=scheduler_variants,
    )
    static = run_condition(
        "static",
        "actions+cot",
        n_episodes=episodes,
        scheduler_variants=scheduler_variants,
    )
    delta = paired_descriptive_difference(static.no_leak, adaptive.no_leak)
    console.print(f"  no-leak fraction, static - adaptive: {delta}")
    console.print(
        "  [dim]The policies differ by an explicit escalation rule. This execution "
        "checks that the authored branch is wired into the simulator; it is not "
        "evidence that a model learns from feedback.[/dim]"
    )
    return [adaptive, static]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedules",
        "--seeds",
        dest="scheduler_variants",
        type=int,
        default=20,
        help="authored task/path scheduler variants (not independent model seeds)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--out", type=Path, default=Path("results/scripted_mechanism_diagnostic.jsonl")
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="write explicitly labeled descriptive diagnostic plots",
    )
    args = parser.parse_args()

    console.print(
        "[bold yellow]SCRIPTED MECHANISM DIAGNOSTIC[/bold yellow] — descriptive only; "
        "no model-behavior, causal, significance, survival, or validated-detection claim"
    )
    feedback = feedback_mechanism_diagnostic(args.scheduler_variants, args.episodes)
    memory = monitor_mechanism_diagnostic(args.scheduler_variants, args.episodes)
    conditions = (
        feedback + memory + hard_coded_policy_contrast(args.scheduler_variants, args.episodes)
    )

    if args.figures is not None:
        from sandbox.figures import plot_scripted_feedback, plot_scripted_tradeoff

        args.figures.mkdir(parents=True, exist_ok=True)
        plot_scripted_feedback(feedback, args.figures / "scripted_feedback_diagnostic.png")

        monitors = [
            "none",
            "actions",
            "actions+cot",
            "window1",
            "window2",
            "window4",
            "cross-episode",
        ]
        rows = []
        for monitor in monitors:
            attack = next(c for c in memory if c.monitor == monitor and c.agent == "adaptive")
            honest = next(c for c in memory if c.monitor == monitor and c.agent == "honest")
            rows.append(
                (
                    monitor,
                    attack.summary().no_leak_fraction.require_mean(),
                    honest.summary().task_completion_fraction.require_mean(),
                )
            )
        plot_scripted_tradeoff(rows, args.figures / "scripted_tradeoff_diagnostic.png")
        console.print(f"wrote 2 descriptive diagnostic figures to {args.figures}")

    save(conditions, args.out)
    console.print(f"\nwrote {len(conditions)} scripted diagnostic conditions to {args.out}")


if __name__ == "__main__":
    main()
