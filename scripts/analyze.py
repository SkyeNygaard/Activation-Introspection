"""Re-analyse a saved sweep without touching a model.

Sweeps are expensive and analysis choices are not final, so every trial is
persisted and every number in the summary is recomputed from JSONL. If a metric
here changes, no GPU time is needed to see the effect on past runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from introspect.analysis import headline, summarize_all
from introspect.experiment import TrialSet
from introspect.metrics import accuracy, bootstrap_auroc

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?", default=Path("results/ladder.jsonl"))
    args = ap.parse_args()

    ts = TrialSet.load(args.path)
    console.print(f"{len(ts.trials)} trials from {args.path}")

    summaries = summarize_all(ts)

    console.rule("per cell")
    table = Table(
        "model",
        "layer",
        "alpha",
        "n",
        "KL",
        "det AUROC",
        "null AUROC",
        "identify",
        "observer",
        "gap",
        "free-form",
    )
    for s in summaries:
        table.add_row(
            s.model.split("/")[-1],
            str(s.layer),
            f"{s.strength}",
            str(s.n),
            f"{s.behavioural_kl.value:.3f}",
            f"{s.detection_auroc.value:.2f}",
            f"{s.detection_auroc_null.value:.2f}",
            f"{s.identify_acc.value:.2f}",
            f"{s.observer_acc.value:.2f}",
            f"{s.gap.value:+.2f}[{s.gap.lo:+.2f},{s.gap.hi:+.2f}]",
            f"{s.free_form_acc.value:.2f}",
        )
    console.print(table)

    console.rule("pooled per model")
    pooled = Table(
        "model", "n", "det AUROC (concept)", "det AUROC (null)", "identify", "observer", "gap"
    )
    for model in sorted({t.model for t in ts.trials}):
        concept = [t for t in ts.trials if t.model == model and t.arm == "concept"]
        clean = [t for t in ts.trials if t.model == model and t.arm == "clean"]
        shuffled = [t for t in ts.trials if t.model == model and t.arm == "shuffled"]
        from introspect.metrics import paired_difference

        gap = paired_difference(
            [float(t.identify_correct) for t in concept],
            [float(t.observer_correct) for t in concept],
        )
        pooled.add_row(
            model.split("/")[-1],
            str(len(concept)),
            str(
                bootstrap_auroc(
                    [t.detection_score for t in concept], [t.detection_score for t in clean]
                )
            ),
            str(
                bootstrap_auroc(
                    [t.detection_score for t in shuffled], [t.detection_score for t in clean]
                )
            ),
            str(accuracy([t.identify_correct for t in concept])),
            str(accuracy([t.observer_correct for t in concept])),
            str(gap),
        )
    console.print(pooled)

    console.print(f"\n[bold]{headline(summaries)}[/bold]")
    console.print(
        "\n[dim]Chance for identification is 1/8 = 0.125. The null AUROC column is the "
        "gate: if the shuffled control detects above chance, the model is responding to "
        "perturbation rather than content and the concept column means nothing.[/dim]"
    )


if __name__ == "__main__":
    main()
