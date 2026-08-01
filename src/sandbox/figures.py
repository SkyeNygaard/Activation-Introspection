"""Figures for the two claims this sandbox supports.

House rule: no plot of a defence without its usefulness cost on the same axes.
A safety number shown alone is the visual form of the overclaim the environment
was built to avoid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sandbox.runner import Condition

BLUE = "#1b6ca8"
ORANGE = "#e08a1e"
GREY = "#9a9a9a"


def plot_feedback(conditions: list[Condition], path: Path) -> None:
    """Attack success vs how much the refusal leaks, with the static control.

    The claim: optimisation pressure against a monitor does not require training
    against it. Informative refusals supply it on their own.
    """
    order = ["silent", "generic", "specific"]
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for agent, colour, marker in (("adaptive", BLUE, "o"), ("static", GREY, "s")):
        xs, ys, los, his = [], [], [], []
        for i, fb in enumerate(order):
            match = [c for c in conditions if c.feedback == fb and c.agent == agent]
            if not match:
                continue
            est = match[0].summary()["safety"]
            xs.append(i)
            ys.append(est.value)
            los.append(est.value - est.lo)
            his.append(est.hi - est.value)
        ax.errorbar(
            xs,
            ys,
            yerr=[los, his],
            marker=marker,
            capsize=4,
            label=f"{agent} attacker",
            color=colour,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        ["silent\n(looks like an error)", 'generic\n("blocked")', "specific\n(names the reason)"],
        fontsize=8,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("what a blocked action tells the agent")
    ax.set_ylabel("safety (1 = credential never leaked)")
    ax.set_title(
        "Attack success is monotone in refusal informativeness\n(no gradient step anywhere)"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frontier(rows: list[tuple[str, float, float, float, float]], path: Path) -> None:
    """Safety against honest usefulness, one point per monitor.

    ``rows`` is (label, safety, safety_err, usefulness, usefulness_err).
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    # Several monitors land on exactly (1.0, 0.0) -- perfectly useful, perfectly
    # unsafe -- so labels are staggered by how many points already share the spot.
    seen: dict[tuple[float, float], int] = {}
    for label, safety, s_err, useful, u_err in rows:
        ax.errorbar(useful, safety, xerr=u_err, yerr=s_err, marker="o", capsize=3, color=BLUE)
        key = (round(useful, 2), round(safety, 2))
        rank = seen.get(key, 0)
        seen[key] = rank + 1
        ax.annotate(
            label,
            (useful, safety),
            textcoords="offset points",
            xytext=(-64 if useful > 0.9 else 6, 5 + rank * 11),
            fontsize=8,
        )
    ax.set_xlabel("usefulness on honest work")
    ax.set_ylabel("safety against the adaptive attacker")
    ax.set_xlim(0.3, 1.08)
    ax.set_ylim(-0.08, 1.12)
    ax.set_title(
        "There is no free defence\n"
        "(the only monitor that stops the attack costs half of honest work)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_leak_delay(rows: list[tuple[str, float, float, float]], path: Path) -> None:
    """Leak episode by monitor memory. ``rows`` is (label, mean, lo, hi)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [r[0] for r in rows]
    xs = range(len(rows))
    ax.errorbar(
        list(xs),
        [r[1] for r in rows],
        yerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
        marker="o",
        capsize=4,
        color=BLUE,
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("episode of first leak (censored at run length)")
    ax.set_title("A bounded monitor window only buys delay\n(the attacker waits it out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_llm_adaptation(json_path: Path, path: Path) -> None:
    """Mean rung by feedback level across model sizes, with the null band.

    The safety-relevant reading: if the lines are flat, small models do not do
    in-context obfuscation from refusal feedback, and the scripted headline is an
    upper bound rather than a description.
    """
    import json

    data = json.loads(json_path.read_text())
    levels = ["silent", "generic", "specific"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colours = {"qwen-0.5b": "#9a9a9a", "qwen-1.5b": ORANGE, "qwen-3b": BLUE}

    for mdl, d in data.items():
        ys = [d["levels"][lv][0] for lv in levels]
        lo = [d["levels"][lv][0] - d["levels"][lv][1] for lv in levels]
        hi = [d["levels"][lv][2] - d["levels"][lv][0] for lv in levels]
        ax.errorbar(
            range(len(levels)),
            ys,
            yerr=[lo, hi],
            marker="o",
            capsize=4,
            label=mdl.replace("qwen-", "Qwen "),
            color=colours.get(mdl, "#333"),
        )

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(["silent", "generic", "specific"], fontsize=9)
    ax.set_xlabel("refusal disclosure")
    ax.set_ylabel("mean rung chosen  (0 = blatant, 3 = subtle)")
    ax.set_ylim(-0.2, 3.2)
    ax.set_title(
        "Does a real model steer toward subtler actions after a refusal?\n"
        "(flat lines = no in-context obfuscation at this scale)",
        fontsize=10,
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
