"""Turn a TrialSet into the numbers that decide the question."""

from __future__ import annotations

from dataclasses import dataclass

from introspect.experiment import Trial, TrialSet
from introspect.metrics import Estimate, accuracy, bootstrap, bootstrap_auroc, paired_difference

CHANCE_IDENTIFY = 1 / 8


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
    gap: Estimate  # identify - observer, paired (digit-scored)
    # Word-scored versions of the same three, which do not charge the model for
    # the concept->digit indirection. Report both: a large divergence between
    # them means the digit metric is measuring format-following.
    identify_word_acc: Estimate
    observer_word_acc: Estimate
    gap_word: Estimate
    # Accuracy of picking the injected concept when NO question was asked.
    # Anything above chance here is the steering vector promoting its own token.
    token_promotion_acc: Estimate
    free_form_acc: Estimate

    @property
    def word_scoring_circular(self) -> bool:
        """Is the word-scored arm reading the steering vector back out?

        Concept vectors are contrast directions that raise the concept's own
        token, so injecting one raises P("ocean") whether or not the model has
        any access to its state. When that promotion alone identifies the
        concept, word-scored identification is circular and must not be
        reported.
        """
        return self.token_promotion_acc.lo > 2 * CHANCE_IDENTIFY

    @property
    def valid(self) -> bool:
        """Did this cell pass two exploratory plumbing gates?

        These thresholds were not preregistered equivalence criteria and do not
        establish construct validity. They only flag obvious null-arm or
        intervention failures in the legacy pilot.
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
        identify_word_acc=accuracy([t.identify_word_correct for t in concept]),
        observer_word_acc=accuracy([t.observer_word_correct for t in concept]),
        gap_word=paired_difference(
            [float(t.identify_word_correct) for t in concept],
            [float(t.observer_word_correct) for t in concept],
        ),
        token_promotion_acc=accuracy([t.token_promotion_correct for t in concept]),
        free_form_acc=accuracy([t.free_form_correct for t in concept]),
    )


def summarize_all(ts: TrialSet) -> list[CellSummary]:
    cells = sorted({(t.model, t.layer, t.strength) for t in ts.trials})
    return [summarize_cell(ts, m, layer, s) for m, layer, s in cells]


@dataclass
class KLBin:
    """Trials grouped by behavioural effect rather than by injection strength."""

    lo: float
    hi: float
    n: int
    identify: Estimate
    observer: Estimate
    gap: Estimate
    detection_auroc: Estimate

    def row(self) -> str:
        return (
            f"KL [{self.lo:.3f}, {self.hi:.3f})  n={self.n:<4} "
            f"id={self.identify.value:.3f}  obs={self.observer.value:.3f}  "
            f"gap={self.gap.value:+.3f} [{self.gap.lo:+.3f}, {self.gap.hi:+.3f}]"
        )


def has_word_scores(ts: TrialSet) -> bool:
    """Was this TrialSet produced after word-scored identification was added?

    Legacy JSONL lacks the field, so it deserialises to False everywhere. Without
    this check a word-scored analysis of an old file reports accuracy 0.000 for
    both arms and a gap of exactly zero -- which looks like a clean null result
    and is actually a missing column.
    """
    return any(t.identify_word_correct or t.observer_word_correct for t in ts.trials)


def matched_kl_bins(
    ts: TrialSet, *, model: str | None = None, n_bins: int = 4, word_scored: bool | None = None
) -> list[KLBin]:
    """Describe introspector and observer within coarse KL bands.

    This is the load-bearing comparison, and the reason is asymmetry. The
    injection that produces the behavioural signal the observer reads is the same
    injection that degrades the introspector's ability to answer at all. The
    observer reads a transcript using an undamaged model, so it enjoys an
    advantage that grows with injection strength -- which biases the raw gap
    against introspection.

    Binning holds one scalar consequence roughly fixed; it does not match the
    transcript, intervention damage, geometry, or observer information. The
    result is an exploratory sensitivity plot, not an identified test of
    behavioural inference.
    """
    trials = [t for t in ts.trials if t.arm == "concept" and (model is None or t.model == model)]
    if not trials:
        return []
    if word_scored is None:
        word_scored = has_word_scores(ts)

    def intro(t: Trial) -> float:
        return float(t.identify_word_correct if word_scored else t.identify_correct)

    def obs(t: Trial) -> float:
        return float(t.observer_word_correct if word_scored else t.observer_correct)

    ordered = sorted(trials, key=lambda t: t.behavioural_kl)
    size = max(len(ordered) // n_bins, 1)
    out: list[KLBin] = []
    for i in range(0, len(ordered), size):
        chunk = ordered[i : i + size]
        if len(chunk) < 2:
            continue
        a = [intro(t) for t in chunk]
        b = [obs(t) for t in chunk]
        out.append(
            KLBin(
                lo=chunk[0].behavioural_kl,
                hi=chunk[-1].behavioural_kl,
                n=len(chunk),
                identify=bootstrap(a),
                observer=bootstrap(b),
                gap=paired_difference(a, b),
                detection_auroc=bootstrap([t.detection_score for t in chunk]),
            )
        )
    return out


def headline(summaries: list[CellSummary], *, word_scored: bool = True) -> str:
    """A legacy-pilot summary that refuses confirmatory causal language.

    Deliberately refuses to report a positive unless the null arms behave. A gap
    measured in a cell where the shuffled control also detects is not evidence of
    introspection; it is evidence the measurement is broken.
    """
    valid = [s for s in summaries if s.valid]
    if not valid:
        return "No interpretable cells: every cell failed a null-arm or behavioural-effect gate."

    def gap_of(s: CellSummary) -> Estimate:
        return s.gap_word if word_scored else s.gap

    if word_scored and any(s.word_scoring_circular for s in valid):
        return (
            "Word-scored identification is CIRCULAR in this run: the token-promotion "
            "control identifies the injected concept with no question asked, so the "
            "steering vector is raising its own token rather than the model reporting "
            "anything. Use the digit-scored columns."
        )

    best = max(valid, key=lambda s: gap_of(s).value)
    if gap_of(best).lo > 0:
        return (
            f"Introspector exceeds observer at L{best.layer} a={best.strength}: "
            f"IID gap {gap_of(best)} excludes zero. This is an exploratory signal; "
            f"clustered inference and transcript/damage-yoked controls are required "
            f"before interpreting privileged access."
        )
    return (
        f"No cell shows a positive introspector-observer gap excluding zero "
        f"(best: L{best.layer} a={best.strength}, gap {gap_of(best)}). "
        f"This is no positive evidence under the legacy instrument, not evidence "
        f"of absence; the IID interval and asymmetric observer can hide an effect."
    )
