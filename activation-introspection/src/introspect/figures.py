"""Legacy exploratory figures with explicit stress-test arms.

No figure shows the concept arm without its null arm, and no legacy report plot
shows the introspector without the observer. Those arms expose some artifacts but
do not make the IID intervals or asymmetric observer confirmatory. The retired
IFT/probe entry point fails loudly rather than regenerating the retracted plot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt

from introspect.analysis import CellSummary, KLBin
from introspect.experiment import Trial, TrialSet
from introspect.metrics import accuracy

CHANCE_IDENTIFY = 1 / 8


def _err(est) -> list[list[float]]:  # type: ignore[no-untyped-def]
    return [[est.value - est.lo], [est.hi - est.value]]


def plot_detection_by_layer(summaries: list[CellSummary], path: Path) -> None:
    """Detection AUROC vs layer, concept arm against the matched-norm null."""
    layers = sorted({s.layer for s in summaries})
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for arm, attr, colour in (
        ("concept", "detection_auroc", "#1b6ca8"),
        ("shuffled null", "detection_auroc_null", "#b0b0b0"),
    ):
        xs, ys, los, his = [], [], [], []
        for layer in layers:
            cells = [s for s in summaries if s.layer == layer]
            if not cells:
                continue
            est = max(cells, key=lambda s: s.n).__getattribute__(attr)
            xs.append(layer)
            ys.append(est.value)
            los.append(est.value - est.lo)
            his.append(est.hi - est.value)
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=arm, color=colour)

    ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.5)
    ax.annotate("chance", (layers[0], 0.5), textcoords="offset points", xytext=(0, 5), fontsize=8)
    ax.set_xlabel("injection layer")
    ax.set_ylabel("detection AUROC (injected vs clean)")
    ax.set_title("Detection: can the model tell something was injected?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_introspector_vs_observer(
    summaries: list[CellSummary], path: Path, *, word_scored: bool = True
) -> None:
    """The headline comparison, per cell, with chance marked."""
    intro_attr = "identify_word_acc" if word_scored else "identify_acc"
    obs_attr = "observer_word_acc" if word_scored else "observer_acc"
    intro = [getattr(s, intro_attr) for s in summaries]
    obs = [getattr(s, obs_attr) for s in summaries]
    labels = [f"L{s.layer}\na={s.strength}" for s in summaries]
    x = range(len(summaries))
    fig, ax = plt.subplots(figsize=(max(7, len(summaries) * 0.9), 4.4))

    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        [e.value for e in intro],
        width,
        yerr=[
            [e.value - e.lo for e in intro],
            [e.hi - e.value for e in intro],
        ],
        capsize=3,
        label="introspector (own activations)",
        color="#1b6ca8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [e.value for e in obs],
        width,
        yerr=[
            [e.value - e.lo for e in obs],
            [e.hi - e.value for e in obs],
        ],
        capsize=3,
        label="observer (output only)",
        color="#e08a1e",
    )

    ax.axhline(CHANCE_IDENTIFY, ls="--", lw=1, color="k", alpha=0.5)
    ax.annotate(
        "chance (1/8)",
        (-0.4, CHANCE_IDENTIFY),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=8,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("identification accuracy")
    ax.set_title("Introspection vs behavioural inference\n(legacy asymmetric observer diagnostic)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_matched_kl(bins: list[KLBin], path: Path) -> None:
    """Legacy gap within bands of similar scalar behavioural effect.

    KL binning does not equalize transcript, damage geometry, or reporter state;
    this plot is descriptive and cannot identify privileged access.
    """
    if not bins:
        return
    centres = [(b.lo + b.hi) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(
        centres,
        [b.gap.value for b in bins],
        yerr=[
            [b.gap.value - b.gap.lo for b in bins],
            [b.gap.hi - b.gap.value for b in bins],
        ],
        marker="o",
        capsize=4,
        color="#1b6ca8",
    )
    ax.axhline(0, ls="--", lw=1, color="k", alpha=0.6)
    ax.annotate(
        "zero legacy gap",
        xy=(centres[0], 0),
        textcoords="offset points",
        xytext=(4, 6),
        fontsize=8,
        color="gray",
    )
    # KL is non-negative, so symlog would waste half the axis on a region
    # that cannot contain data.
    ax.set_xscale("log")
    ax.set_xlabel("behavioural effect, KL(clean || intervened)")
    ax.set_ylabel("introspector minus observer (paired)")
    ax.set_title(
        "Gap at matched behavioural effect\n(flat at zero = behavioural inference suffices)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scale_ladder(ts: TrialSet, path: Path, *, word_scored: bool = True) -> None:
    """Does the gap open up with model size? The question a single model cannot answer."""
    models = sorted({t.model for t in ts.trials}, key=lambda m: len(m))
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))

    def intro_of(t: Trial) -> bool:
        return t.identify_word_correct if word_scored else t.identify_correct

    def obs_of(t: Trial) -> bool:
        return t.observer_word_correct if word_scored else t.observer_correct

    series: list[tuple[str, Callable[[Trial], bool], str]] = [
        ("introspector", intro_of, "#1b6ca8"),
        ("observer", obs_of, "#e08a1e"),
    ]
    for label, getter, colour in series:
        ys, los, his = [], [], []
        for m in models:
            trials = [t for t in ts.trials if t.model == m and t.arm == "concept"]
            est = accuracy([getter(t) for t in trials])
            ys.append(est.value)
            los.append(est.value - est.lo)
            his.append(est.hi - est.value)
        ax.errorbar(
            range(len(models)),
            ys,
            yerr=[los, his],
            marker="o",
            capsize=4,
            label=label,
            color=colour,
        )

    ax.axhline(CHANCE_IDENTIFY, ls="--", lw=1, color="k", alpha=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1].replace("-Instruct", "") for m in models], fontsize=8)
    ax.set_ylabel("identification accuracy")
    ax.set_title("Scale ladder")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(
    ts: TrialSet,
    summaries: list[CellSummary],
    bins: list[KLBin],
    outdir: Path,
    *,
    word_scored: bool = True,
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    plot_detection_by_layer(summaries, outdir / "detection_by_layer.png")
    plot_introspector_vs_observer(
        summaries, outdir / "introspector_vs_observer.png", word_scored=word_scored
    )
    plot_matched_kl(bins, outdir / "matched_kl_gap.png")
    plot_scale_ladder(ts, outdir / "scale_ladder.png", word_scored=word_scored)
    for name in (
        "detection_by_layer.png",
        "introspector_vs_observer.png",
        "matched_kl_gap.png",
        "scale_ladder.png",
    ):
        if (outdir / name).exists():
            written.append(outdir / name)
    return written


def plot_ift_vs_probe(ift_json: Path, path: Path) -> None:
    """Retired: the legacy inputs used mismatched sites and invalid IID bars."""
    del ift_json, path
    raise RuntimeError(
        "legacy IFT-versus-probe figure retired: regenerate a schema-2 matched-site "
        "profile and raw multi-run IFT artifact before plotting a descriptive join"
    )
