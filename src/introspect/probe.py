"""Is the injected concept *present but unreportable*, or genuinely absent?

## The gap this targets

Two 2026 results sit in tension:

- Binary detection of an injection is entirely explained by a global logit shift
  (Krasheninnikov-style control: the same shift appears on factually-false
  questions, r=0.999). Detection is confounded.
- Yet models identify *which sentence* was perturbed at ~88% against 10% chance,
  while confabulating when asked *what* was injected -- guesses drift toward
  common concrete nouns ("apple").

So a model can localize a perturbation it cannot name. That leaves an unanswered
question with a direct bearing on whether verbalization training can work:

    Is the concept's identity absent from the state the model answers from,
    or present in it and simply not reaching the output?

## The measurement

For each trial, capture the residual stream at **the exact position the model
generates its answer from** -- the final token of the identification prompt,
under injection. Then fit a linear probe to predict which concept was injected,
and compare probe accuracy to the model's own answer on held-out trials.

- probe ≈ self-report → the limit is representational. Nothing to verbalize.
- probe ≫ self-report → the identity is linearly available at the decision point
  and the model still fails to say it. An **access** limit, not a representation
  limit, which is the case where verbalization training has headroom.

## The circularity trap, and the only control that escapes it

**A raw probe on injected activations is meaningless.** Measured here: probe
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

Three quantities are therefore reported together, and only the third means
anything:

1. ``probe_acc_raw`` -- trivially ~1.0. Included as a sanity check that the
   injection landed, and as a standing reminder of the trap.
2. ``probe_acc_shuffled`` -- label-permuted null.
3. ``probe_acc_ablated`` -- **the measurement**. Identity decodable after the
   injected direction is projected out.

Interpretation of the third against self-report:

- ablated probe ~ chance → the model never elaborates the concept; there is
  nothing beyond the injected vector for introspection to reach, and the limit is
  representational. Verbalization training would have nothing to latch onto.
- ablated probe >> self-report → the model builds a content-specific
  representation it does not report. An **access** limit, and the case where
  verbalization training has headroom.

Even then the claim stays narrow: linear decodability of an elaborated
representation is not "the model knows". It bounds headroom, nothing more.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from introspect.concepts import ConceptVector
from introspect.grading import score_choices
from introspect.hooks import Intervention, capture, intervene
from introspect.metrics import Estimate, accuracy
from introspect.models import LoadedModel
from introspect.prompts import IDENTIFY_FORCED_CHOICE_VARIANTS, forced_choice, variant


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
                f"After projecting out the injected direction, concept identity is NOT "
                f"decodable ({self.probe_acc_ablated} vs null "
                f"{self.probe_acc_shuffled}). The model does not elaborate the "
                f"injection into a content-specific representation, so there is nothing "
                f"beyond the injected vector for introspection to reach. This is a "
                f"REPRESENTATION limit -- verbalization training has nothing to latch "
                f"onto at this scale. (Raw probe {self.probe_acc_raw.value:.2f} is the "
                f"circular measurement and means nothing.)"
            )
        if self.gap.lo > 0:
            return (
                f"Concept identity survives ablation of the injected direction at "
                f"{self.probe_acc_ablated.value:.2f} (null "
                f"{self.probe_acc_shuffled.value:.2f}) while the model's own answer is "
                f"at {self.self_report_acc.value:.2f} (chance {self.chance:.3f}). The "
                f"model builds a representation it does not report: an ACCESS limit, "
                f"and verbalization training has headroom."
            )
        return (
            f"Ablated probe {self.probe_acc_ablated.value:.2f} vs self-report "
            f"{self.self_report_acc.value:.2f}: no reliable gap."
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
    option_block = forced_choice(options)
    digits = [str(i + 1) for i in range(len(options))]

    feats: list[np.ndarray] = []
    ablated: list[np.ndarray] = []
    labels: list[int] = []
    correct: list[bool] = []

    for concept_idx, name in enumerate(options):
        vec = bank[name]
        for seed in seeds:
            prompt = model.chat(
                variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(options=option_block)
            )
            ids = model.encode(prompt)
            iv = Intervention(layer=inject_layer, direction=vec.vector, strength=strength)

            # Register the intervention first so capture observes the edited stream.
            with intervene(model, [iv], prompt_len=int(ids.shape[1])):
                with capture(model, [probe_layer]) as store:
                    model.forward_logits(ids)
                choice = score_choices(model, prompt, digits, interventions=[iv])

            act = store.last_token(probe_layer)[0]
            feats.append(act.numpy())

            # Project out the exact direction that was injected. Whatever
            # identity survives this is the model's own elaboration.
            unit = vec.vector / (vec.vector.norm() + 1e-8)
            ablated.append((act - torch.dot(act, unit) * unit).numpy())

            labels.append(concept_idx)
            correct.append(choice.argmax == concept_idx)

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

NATURAL_TEMPLATES = [
    "The documentary explored how {concept} shapes daily life in the region.",
    "She wrote three pages about {concept} before dinner.",
    "Most people underestimate how complicated {concept} really is.",
    "There was a long article on {concept} in the weekend paper.",
    "He kept returning to the subject of {concept} all evening.",
    "A short lecture introduced the history of {concept}.",
]


@torch.no_grad()
def collect_natural(
    model: LoadedModel, concepts: Sequence[str], layer: int
) -> tuple[np.ndarray, np.ndarray]:
    """Activations from ordinary text mentioning each concept. No injection.

    This is the training distribution for the transfer probe. Nothing here has
    been perturbed, so a decision boundary fit on it encodes what the model's own
    representation of each concept looks like.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for idx, name in enumerate(concepts):
        for template in NATURAL_TEMPLATES:
            ids = model.encode(template.format(concept=name))
            with capture(model, [layer]) as store:
                model.forward_logits(ids)
            feats.append(store.last_token(layer)[0].numpy())
            labels.append(idx)
    return np.stack(feats), np.array(labels)


@dataclass
class TransferResult:
    layer: int
    n_train: int
    n_test: int
    n_classes: int
    transfer_acc: Estimate  # natural-trained probe, tested on injected trials
    transfer_acc_shuffled: Estimate
    within_natural_acc: Estimate  # does the probe work at all on its own domain?
    self_report_acc: Estimate

    @property
    def chance(self) -> float:
        return 1.0 / self.n_classes

    @property
    def verdict(self) -> str:
        if self.within_natural_acc.lo < 2 * self.chance:
            return (
                f"The probe cannot separate these concepts even in natural text "
                f"({self.within_natural_acc}); nothing downstream is interpretable."
            )
        if self.transfer_acc.lo <= self.transfer_acc_shuffled.hi:
            return (
                f"A probe that reads natural concept representations at "
                f"{self.within_natural_acc.value:.2f} does NOT transfer to injected "
                f"trials ({self.transfer_acc} vs null {self.transfer_acc_shuffled}). "
                f"The injection does not put the model into a state resembling its own "
                f"representation of the concept -- so there is no concept-like content "
                f"for introspection to report, and self-report at "
                f"{self.self_report_acc.value:.2f} is exactly what that predicts."
            )
        return (
            f"A natural-text probe transfers to injected trials at {self.transfer_acc} "
            f"(null {self.transfer_acc_shuffled}) while the model reports at "
            f"{self.self_report_acc.value:.2f} (chance {self.chance:.3f}). The injection "
            f"induces a genuinely concept-like state that the model does not verbalize: "
            f"headroom for verbalization training."
        )


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

    This is the design that answers the question, because the probe never sees an
    injected activation during training. It cannot recover "the vector we added";
    it can only succeed if injection moves the model toward the same region of
    representation space that genuinely thinking about the concept does.
    """
    concepts = sorted(bank)
    x_nat, y_nat = collect_natural(model, concepts, probe_layer)
    _, x_inj, y_inj, self_report = collect(
        model, bank, inject_layer, probe_layer, strength, seeds=seeds
    )

    scaler = StandardScaler().fit(x_nat)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(x_nat), y_nat)
    transfer = list(clf.predict(scaler.transform(x_inj)) == y_inj)

    rng = np.random.default_rng(0)
    y_shuf = y_inj.copy()
    rng.shuffle(y_shuf)
    transfer_null = list(clf.predict(scaler.transform(x_inj)) == y_shuf)

    return TransferResult(
        layer=probe_layer,
        n_train=len(y_nat),
        n_test=len(y_inj),
        n_classes=len(concepts),
        transfer_acc=accuracy(transfer),
        transfer_acc_shuffled=accuracy(transfer_null),
        within_natural_acc=accuracy(fit_probe(x_nat, y_nat, n_splits=3)),
        self_report_acc=accuracy(self_report),
    )
