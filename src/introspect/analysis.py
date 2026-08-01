"""Turn a TrialSet into the numbers that decide the question."""

from __future__ import annotations

from dataclasses import dataclass

from introspect.experiment import Trial, TrialSet
from introspect.metrics import Estimate, accuracy, bootstrap, bootstrap_auroc, paired_difference


@dataclass
class CellSummary:
    """Everything reportable about one (model, layer, strength) cell."""

    model: str
    layer: int
    strength: float
    n: int

    behavioural_kl: Estimate
    detection_auroc: Estimate  # concept vs clean
    detection_auroc_null: Estimate  # shuffled vs clean -- must be ~0.5
    identify_acc: Estimate
    identify_acc_null: Estimate
    observer_acc: Estimate
    gap: Estimate  # identify - observer, paired
    free_form_acc: Estimate

    @property
    def valid(self) -> bool:
        """Is this cell interpretable at all?

        Two gates. If the null arm detects above chance, the model is responding
        to *perturbation* rather than content and the concept arm means nothing.
        If the behavioural effect is zero, nothing was injected and a null result
        is trivial.
        """
        return self.detection_auroc_null.lo < 0.65 and self.behavioural_kl.value > 1e-4

    def row(self) -> str:
        flag = "" if self.valid else "  [INVALID]"
        return (
            f"L{self.layer:<3} a={self.strength:<5} KL={self.behavioural_kl.value:7.4f}  "
            f"detAUROC={self.detection_auroc.value:.2f}"
            f"(null {self.detection_auroc_null.value:.2f})  "
            f"id={self.identify_acc.value:.2f}  obs={self.observer_acc.value:.2f}  "
            f"gap={self.gap.value:+.3f} [{self.gap.lo:+.2f},{self.gap.hi:+.2f}]{flag}"
        )


def _key(t: Trial) -> tuple[str, int]:
    return (t.concept, t.seed)


def _paired(trials_a: list[Trial], trials_b: list[Trial]) -> tuple[list[float], list[float]]:
    """Align introspector and observer trial-by-trial on (concept, seed).

    Both quantities live on the same Trial, but aligning explicitly keeps the
    pairing correct if the arms are ever split across records.
    """
    b_by_key = {_key(t): t for t in trials_b}
    a_vals: list[float] = []
    b_vals: list[float] = []
    for t in trials_a:
        if _key(t) in b_by_key:
            a_vals.append(float(t.identify_correct))
            b_vals.append(float(b_by_key[_key(t)].observer_correct))
    return a_vals, b_vals


def summarize_cell(ts: TrialSet, model: str, layer: int, strength: float) -> CellSummary:
    def sel(arm: str) -> list[Trial]:
        return [
            t
            for t in ts.trials
            if t.model == model and t.layer == layer and t.strength == strength and t.arm == arm
        ]

    concept, clean, shuffled = sel("concept"), sel("clean"), sel("shuffled")

    a_vals, b_vals = _paired(concept, concept)

    return CellSummary(
        model=model,
        layer=layer,
        strength=strength,
        n=len(concept),
        behavioural_kl=bootstrap([t.behavioural_kl for t in concept]),
        detection_auroc=bootstrap_auroc(
            [t.detection_score for t in concept], [t.detection_score for t in clean]
        ),
        detection_auroc_null=bootstrap_auroc(
            [t.detection_score for t in shuffled], [t.detection_score for t in clean]
        ),
        identify_acc=accuracy([t.identify_correct for t in concept]),
        identify_acc_null=accuracy([t.identify_correct for t in shuffled]),
        observer_acc=accuracy([t.observer_correct for t in concept]),
        gap=paired_difference(a_vals, b_vals),
        free_form_acc=accuracy([t.free_form_correct for t in concept]),
    )


def summarize_all(ts: TrialSet) -> list[CellSummary]:
    cells = sorted({(t.model, t.layer, t.strength) for t in ts.trials})
    return [summarize_cell(ts, m, layer, s) for m, layer, s in cells]


def headline(summaries: list[CellSummary]) -> str:
    """The one-paragraph verdict, stated conservatively.

    Deliberately refuses to report a positive unless the null arms behave. A gap
    measured in a cell where the shuffled control also detects is not evidence of
    introspection; it is evidence the measurement is broken.
    """
    valid = [s for s in summaries if s.valid]
    if not valid:
        return "No interpretable cells: every cell failed a null-arm or behavioural-effect gate."

    best = max(valid, key=lambda s: s.gap.value)
    if best.gap.lo > 0:
        return (
            f"Introspector exceeds observer at L{best.layer} a={best.strength}: "
            f"gap {best.gap} excludes zero. Consistent with privileged access at this cell."
        )
    return (
        f"No cell shows an introspector-observer gap excluding zero "
        f"(best: L{best.layer} a={best.strength}, gap {best.gap}). "
        f"Behavioural inference explains the reports at every strength tested."
    )
