"""What concept information is linearly recoverable after an intervention?

## The gap this targets

Two 2026 results sit in tension:

- Binary detection of an injection is entirely explained by a global logit shift
  (Krasheninnikov-style control: the same shift appears on factually-false
  questions, r=0.999). Detection is confounded.
- Yet models identify *which sentence* was perturbed at ~88% against 10% chance,
  while confabulating when asked *what* was injected -- guesses drift toward
  common concrete nouns ("apple").

So a model can localize a perturbation it does not name under one elicitation.
That motivates a narrower measurement question:

    Is the concept's identity absent from the state the model answers from,
    or present in it and simply not reaching the output?

## The measurement

For each trial, capture the residual stream at **the exact position the model
generates its answer from** -- the final token of the identification prompt,
under injection. Then fit a linear probe to predict which concept was injected,
and compare probe accuracy to the model's own answer on held-out trials.

- probe ≈ self-report → no dissociation is detected by these instruments.
- probe ≫ self-report → identity is linearly recoverable at the decision point
  while this report format performs worse. This is a decodability/report
  dissociation, not by itself an access mechanism or evidence that training will
  help.

## The circularity trap, and the only control that escapes it

**A raw probe on injected activations is circular as evidence of model
computation.** Measured here: probe
accuracy was exactly 1.000 at every layer probed (11, 16, 23) with an injection
at layer 9, against a 0.113 shuffled null. That is not a finding. The residual
stream is additive, so an injected direction propagates forward essentially
verbatim; a linear probe recovering it has recovered the thing we added. Probing
"downstream" does not help, which is why the number is 1.000 at the last layer
too.

The escape is to **project the injected direction out of the representation and
probe the remainder**:

    a_residual = a - (a . u) u        where u is the unit injected direction

If concept identity is still decodable after the exact injected vector is
removed, then the model's own computation has elaborated the concept into
directions we did not put there. That is a claim about the model, not about our
own arithmetic.

Three quantities are therefore reported together. The third is less direct than
the raw probe, but remains a diagnostic rather than a proof of elaboration:

1. ``probe_acc_raw`` -- trivially ~1.0. Included as a sanity check that the
   injection landed, and as a standing reminder of the trap.
2. ``probe_acc_shuffled`` -- label-permuted null.
3. ``probe_acc_ablated`` -- **the measurement**. Identity decodable after the
   injected direction is projected out.

Conservative interpretation of the third against self-report:

- ablated probe ~ chance → this probe finds no residual signal after the chosen
  projection. That can reflect absent signal, an incomplete readout, limited data,
  or a representation not captured by this linear control.
- ablated probe >> self-report → this pipeline finds a residual linear signal
  that the selected report endpoint does not express. Format competence,
  intervention damage, probe leakage, and held-out transfer still have to be
  ruled out.

Even then the claim stays narrow: linear decodability is not "the model knows,"
does not establish causal use, and does not by itself bound training headroom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from introspect.concepts import ConceptVector
from introspect.grading import score_choices
from introspect.hooks import Intervention, capture, intervene
from introspect.metrics import Estimate, accuracy
from introspect.models import LoadedModel
from introspect.prompts import (
    IDENTIFY_FORCED_CHOICE_VARIANTS,
    forced_choice,
    permuted_options,
    variant,
)


@dataclass
class ProbeResult:
    layer_injected: int
    layer_probed: int
    n: int
    n_classes: int
    probe_acc_raw: Estimate  # trivially ~1.0; the trap, kept visible
    probe_acc_ablated: Estimate  # THE measurement: injected direction removed
    probe_acc_shuffled: Estimate  # label-permuted null
    self_report_acc: Estimate
    gap: Estimate  # ablated probe - self report

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def verdict(self) -> str:
        if self.probe_acc_ablated.hi <= self.probe_acc_shuffled.hi:
            return (
                f"After projecting out the injected direction, this probe does not "
                f"separate concept identity from its permuted-label diagnostic "
                f"({self.probe_acc_ablated} vs {self.probe_acc_shuffled}). This is "
                f"inconclusive about whether signal is absent, nonlinear, or "
                f"underpowered. Raw probe {self.probe_acc_raw.value:.2f} is expected "
                f"to recover the injected axis and is not evidence of elaboration."
            )
        if self.gap.lo > 0:
            return (
                f"Concept identity survives ablation of the injected direction at "
                f"{self.probe_acc_ablated.value:.2f} (null "
                f"{self.probe_acc_shuffled.value:.2f}) while the model's own answer is "
                f"at {self.self_report_acc.value:.2f} (chance {self.chance:.3f}). This "
                f"is an exploratory linear-decoding/report dissociation under an IID "
                f"interval, not proof of privileged access or training headroom."
            )
        return (
            f"Ablated probe {self.probe_acc_ablated.value:.2f} vs self-report "
            f"{self.self_report_acc.value:.2f}: no positive dissociation under this "
            f"instrument; clustered confirmatory inference was not run."
        )


@torch.no_grad()
def collect(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    inject_layer: int,
    probe_layer: int,
    strength: float,
    *,
    seeds: Sequence[int] = range(8),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Return (raw_acts, ablated_acts, labels, self_report_correct).

    Activations are taken at the final position of the identification prompt --
    precisely the state from which the next token (the model's answer) is drawn.
    ``ablated_acts`` has the injected direction projected out; it is the only one
    that supports a claim about the model.
    """
    options = sorted(bank)
    digits = [str(i + 1) for i in range(len(options))]

    feats: list[np.ndarray] = []
    ablated: list[np.ndarray] = []
    labels: list[int] = []
    correct: list[bool] = []

    for concept_idx, name in enumerate(options):
        vec = bank[name]
        for seed in seeds:
            trial_options = permuted_options(options, seed)
            prompt = model.chat(
                variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(
                    options=forced_choice(trial_options)
                )
            )
            ids = model.encode(prompt)
            iv = Intervention(layer=inject_layer, direction=vec.vector, strength=strength)

            # One intervened forward supplies both capture and answer logits.
            # Passing ``iv`` to score_choices inside the outer context used to
            # register it twice, silently doubling the self-report strength.
            with (
                intervene(model, [iv], prompt_len=int(ids.shape[1])),
                capture(model, [probe_layer]) as store,
            ):
                choice = score_choices(model, prompt, digits)

            act = store.last_token(probe_layer)[0]
            feats.append(act.numpy())

            # Project out the exact direction that was injected. Whatever
            # identity survives this is the model's own elaboration.
            unit = vec.vector / (vec.vector.norm() + 1e-8)
            ablated.append((act - torch.dot(act, unit) * unit).numpy())

            labels.append(concept_idx)
            correct.append(choice.argmax == trial_options.index(name))

    return np.stack(feats), np.stack(ablated), np.array(labels), correct


def fit_probe(
    x: np.ndarray, y: np.ndarray, *, n_splits: int = 5, seed: int = 0, shuffle_labels: bool = False
) -> list[bool]:
    """Cross-validated multinomial logistic probe. Returns per-trial correctness.

    Cross-validation is stratified so every fold sees every concept, and the
    probe never scores a trial it was trained on. ``shuffle_labels`` gives the
    null: a probe that has learned only injection magnitude, or that is
    overfitting, scores at chance here.
    """
    target = y.copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(target)

    out = np.zeros(len(target), dtype=bool)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(x, target):
        scaler = StandardScaler().fit(x[train])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(x[train]), target[train])
        out[test] = clf.predict(scaler.transform(x[test])) == target[test]
    return list(out)


def run_probe(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    inject_layer: int,
    probe_layer: int,
    strength: float,
    *,
    seeds: Sequence[int] = range(8),
) -> ProbeResult:
    x_raw, x_ablated, y, self_report = collect(
        model, bank, inject_layer, probe_layer, strength, seeds=seeds
    )
    raw_correct = fit_probe(x_raw, y)
    ablated_correct = fit_probe(x_ablated, y)
    # The null is fit on the ablated features, since that is the arm it gates.
    shuffled = fit_probe(x_ablated, y, shuffle_labels=True)

    from introspect.metrics import paired_difference

    return ProbeResult(
        layer_injected=inject_layer,
        layer_probed=probe_layer,
        n=len(y),
        n_classes=len(bank),
        probe_acc_raw=accuracy(raw_correct),
        probe_acc_ablated=accuracy(ablated_correct),
        probe_acc_shuffled=accuracy(shuffled),
        self_report_acc=accuracy(self_report),
        gap=paired_difference([float(c) for c in ablated_correct], [float(c) for c in self_report]),
    )


# -- transfer probe: the only design that escapes the trap ---------------------

# Natural sentences mentioning each concept. Deliberately many and varied.
#
# Power: a logistic probe over an 896-dim residual needs far more than a handful
# of examples per class or it separates any labelling by overfitting. At 6 per
# class (the first version of this experiment) the sample-to-dimension ratio was
# 0.007 and the within-natural estimate was worthless. 40 templates gives 40 per
# class, and the probe is scored with GroupKFold on template id so it must
# generalise to sentence frames it never saw.
NATURAL_TEMPLATES = [
    "The documentary explored how {concept} shapes daily life in the region.",
    "She wrote three pages about {concept} before dinner.",
    "Most people underestimate how complicated {concept} really is.",
    "There was a long article on {concept} in the weekend paper.",
    "He kept returning to the subject of {concept} all evening.",
    "A short lecture introduced the history of {concept}.",
    "Nobody in the room had strong opinions about {concept}.",
    "The exhibition devoted an entire wing to {concept}.",
    "Her thesis argued that {concept} had been badly misunderstood.",
    "They argued about {concept} until the restaurant closed.",
    "The podcast episode on {concept} ran for two hours.",
    "I had never given much thought to {concept} before that trip.",
    "The committee's report barely mentioned {concept}.",
    "Children in the village learn about {concept} early.",
    "A grainy photograph of {concept} hung above the desk.",
    "Funding for research into {concept} was cut last year.",
    "The novel opens with a description of {concept}.",
    "Local traditions around {concept} go back centuries.",
    "He collected books about {concept} obsessively.",
    "The museum guide explained the significance of {concept}.",
    "Weather permitting, we will look at {concept} tomorrow.",
    "Her grandmother told stories involving {concept}.",
    "The magazine ran a cover feature on {concept}.",
    "Students find {concept} harder than the syllabus suggests.",
    "An old map marked the location of {concept}.",
    "The professor's specialism was the economics of {concept}.",
    "Conversations kept drifting back toward {concept}.",
    "A documentary crew spent a month filming {concept}.",
    "The archive holds several manuscripts describing {concept}.",
    "Nothing in the briefing prepared them for {concept}.",
    "They named the boat after {concept}.",
    "The lecture hall emptied before he finished on {concept}.",
    "Few industries are as dependent on {concept}.",
    "A grant proposal on {concept} was submitted in March.",
    "The painting is widely read as a study of {concept}.",
    "Regulations covering {concept} changed in the spring.",
    "His childhood was full of {concept}.",
    "The catalogue lists forty entries under {concept}.",
    "Visitors are often surprised by {concept}.",
    "The final chapter returns to {concept}.",
]


@torch.no_grad()
def collect_natural(
    model: LoadedModel, concepts: Sequence[str], layer: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activations from ordinary text mentioning each concept. No injection.

    This is the training distribution for the transfer probe. Nothing here has
    been perturbed, so a decision boundary fit on it encodes what the model's own
    representation of each concept looks like.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []  # template id, so CV can hold out whole sentence frames
    for idx, name in enumerate(concepts):
        for t_id, template in enumerate(NATURAL_TEMPLATES):
            ids = model.encode(template.format(concept=name))
            with capture(model, [layer]) as store:
                model.forward_logits(ids)
            feats.append(store.last_token(layer)[0].numpy())
            labels.append(idx)
            groups.append(t_id)
    return np.stack(feats), np.array(labels), np.array(groups)


@dataclass
class TransferResult:
    layer: int
    n_train: int
    n_test: int
    n_classes: int
    transfer_acc: Estimate  # natural-trained probe, tested on injected trials
    # THE null: retrain on permuted *training* labels, then test on injected.
    # Shuffling test labels only asks "is this above chance"; permuting the
    # training labels asks the stronger question "can this pipeline manufacture
    # apparent signal from noise at this sample size?"
    transfer_acc_permuted: Estimate
    within_natural_acc: Estimate  # grouped CV: unseen sentence frames
    train_acc: Estimate  # in-sample; large gap vs within_natural means overfitting
    self_report_acc: Estimate

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def overfit_gap(self) -> float:
        return self.train_acc.value - self.within_natural_acc.value

    @property
    def verdict(self) -> str:
        if self.transfer_acc.lo <= self.transfer_acc_permuted.hi:
            return (
                f"Transfer ({self.transfer_acc}) does not exceed the permuted-label "
                f"null ({self.transfer_acc_permuted}). At this sample size the "
                f"pipeline can manufacture the effect from noise; the result is not "
                f"interpretable."
            )
        if self.within_natural_acc.lo < 2 * self.chance:
            return (
                f"The probe cannot separate these concepts even in natural text "
                f"({self.within_natural_acc}); transfer results from this readout are "
                f"uninformative rather than evidence of an absent representation."
            )
        return (
            f"A natural-text probe transfers to injected trials at {self.transfer_acc} "
            f"(null {self.transfer_acc_permuted}) while the model reports at "
            f"{self.self_report_acc.value:.2f} (chance {self.chance:.3f}). This is "
            f"consistent with alignment to this natural-text decision boundary; it "
            f"does not establish causal use, introspective access, or training headroom."
        )


def fit_probe_grouped(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[bool]:
    """CV that holds out whole sentence templates.

    Plain k-fold would let the probe see the same sentence frame with a different
    concept in both train and test, which inflates the estimate. Grouping by
    template forces generalisation to frames it has never seen.
    """
    out = np.zeros(len(y), dtype=bool)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train, test in cv.split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        clf = LogisticRegression(max_iter=3000).fit(scaler.transform(x[train]), y[train])
        out[test] = clf.predict(scaler.transform(x[test])) == y[test]
    return list(out)


def run_transfer_probe(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    inject_layer: int,
    probe_layer: int,
    strength: float,
    *,
    seeds: Sequence[int] = range(8),
) -> TransferResult:
    """Train on natural concept text, test on injected trials.

    The probe never sees an injected activation during training, so success shows
    transfer of a natural-text decision boundary to the injected distribution.
    It does not establish that the boundary is causally used, nor rule out direct
    geometric alignment between the constructed vector and that boundary.
    """
    concepts = sorted(bank)
    x_nat, y_nat, g_nat = collect_natural(model, concepts, probe_layer)
    _, x_inj, y_inj, self_report = collect(
        model, bank, inject_layer, probe_layer, strength, seeds=seeds
    )

    scaler = StandardScaler().fit(x_nat)
    xs_nat, xs_inj = scaler.transform(x_nat), scaler.transform(x_inj)

    clf = LogisticRegression(max_iter=3000).fit(xs_nat, y_nat)
    transfer = list(clf.predict(xs_inj) == y_inj)
    train_correct = list(clf.predict(xs_nat) == y_nat)

    # Permuted-label null: retrain the whole pipeline on shuffled training
    # labels and test on injected trials. Averaged over several permutations so
    # the null is itself estimated rather than being one draw.
    rng = np.random.default_rng(0)
    permuted: list[bool] = []
    for _ in range(5):
        y_perm = rng.permutation(y_nat)
        null_clf = LogisticRegression(max_iter=3000).fit(xs_nat, y_perm)
        permuted.extend(list(null_clf.predict(xs_inj) == y_inj))

    return TransferResult(
        layer=probe_layer,
        n_train=len(y_nat),
        n_test=len(y_inj),
        n_classes=len(concepts),
        transfer_acc=accuracy(transfer),
        transfer_acc_permuted=accuracy(permuted),
        within_natural_acc=accuracy(fit_probe_grouped(x_nat, y_nat, g_nat)),
        train_acc=accuracy(train_correct),
        self_report_acc=accuracy(self_report),
    )
