"""Build concept directions from contrast pairs, plus matched control directions.

A concept vector here is the mean difference in residual stream between prompts
that mention a concept and matched prompts that do not. This is the cheapest
construction that works; it is also the one most likely to be confounded by
surface token statistics, which is why every experiment needs the controls in
``random_control`` and ``shuffled_control`` rather than only a no-injection arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from introspect.hooks import capture
from introspect.models import LoadedModel

# Concrete, imageable nouns work best: the model has a strong lexical prior for
# each, so a successful report is unambiguous to grade.
DEFAULT_CONCEPTS = [
    "ocean",
    "bread",
    "volcano",
    "violin",
    "spider",
    "hospital",
    "desert",
    "clock",
]

# Every template ends ON the concept, and every template guarantees at least a
# few tokens of preceding context. Both constraints are load-bearing.
#
# Ending on the concept: with a trailing "." the captured position sits after the
# concept and mostly encodes "an unusual word appeared recently" rather than which
# word.
#
# Requiring context: a bare "{concept}" template is fatal. Single-token fillers
# like "thing" tokenize to one token, so the captured last position IS position 0
# -- the attention sink -- where Qwen2.5-0.5B puts a massive activation (coord 62
# reaches 1537 against a typical residual norm of 20). That one prompt dominated
# the mean over every other template and made all concept vectors collinear at
# cosine 1.00. See MIN_CONTEXT_TOKENS below.
TEMPLATES = [
    "Think about {concept}",
    "The topic is {concept}",
    "Here is a word: {concept}",
    "Picture it clearly in your mind: {concept}",
    "Subject for today: {concept}",
    "The next word is {concept}",
]

NEUTRAL_FILLERS = ["thing", "item", "object"]

# Sequences shorter than this put the capture position at or near the attention
# sink, where residual magnitudes are 1-2 orders of magnitude larger than normal.
MIN_CONTEXT_TOKENS = 3


@dataclass(frozen=True)
class ConceptVector:
    name: str
    layer: int
    vector: Tensor  # [d_model], float32 on CPU

    def unit(self) -> Tensor:
        return cast(Tensor, self.vector / (self.vector.norm() + 1e-8))


class SinkPositionError(ValueError):
    """Raised when a capture would land on or near the attention sink."""


@torch.no_grad()
def _last_token(model: LoadedModel, text: str, layer: int) -> Tensor:
    ids = model.encode(text)
    n_tokens = int(ids.shape[1])
    if n_tokens < MIN_CONTEXT_TOKENS:
        raise SinkPositionError(
            f"{text!r} is {n_tokens} token(s); capturing there reads the attention "
            f"sink, whose magnitude is orders of magnitude above a normal residual "
            f"and which silently dominates any averaged contrast vector. "
            f"Use a template with at least {MIN_CONTEXT_TOKENS} tokens."
        )
    with capture(model, [layer]) as store:
        model.model(ids)
    return store.last_token(layer)[0]


def build_concept_vector(model: LoadedModel, concept: str, layer: int) -> ConceptVector:
    """Mean over *paired* differences: act(template[concept]) - act(template[filler]).

    Pairing by template matters. Taking mean(positives) - mean(negatives) instead
    lets template-specific structure survive the subtraction whenever the two sets
    are not perfectly balanced, and that residue is large enough to swamp the
    concept signal.
    """
    diffs = [
        _last_token(model, t.format(concept=concept), layer)
        - _last_token(model, t.format(concept=filler), layer)
        for t in TEMPLATES
        for filler in NEUTRAL_FILLERS
    ]
    return ConceptVector(name=concept, layer=layer, vector=torch.stack(diffs).mean(0))


def center_bank(bank: dict[str, ConceptVector]) -> dict[str, ConceptVector]:
    """Remove the direction every concept vector shares.

    Residual streams carry a few enormous outlier dimensions -- on Qwen the mean
    activation norm at mid depth is in the hundreds, dominated by a handful of
    coordinates. Any two contrast vectors built from that stream inherit the same
    outlier component and come out at cosine ~1.0 with each other, which makes
    concept *identification* meaningless: injecting "ocean" and injecting "violin"
    would be the same intervention.

    Subtracting the bank mean removes what all content words share and keeps what
    distinguishes them. Always check ``pairwise_cosines`` after building a bank;
    if the off-diagonal is not near zero, no identification result is valid.
    """
    if not bank:
        return {}
    mean = torch.stack([cv.vector for cv in bank.values()]).mean(0)
    return {
        name: ConceptVector(name=name, layer=cv.layer, vector=cv.vector - mean)
        for name, cv in bank.items()
    }


def build_bank(
    model: LoadedModel,
    layer: int,
    concepts: list[str] | None = None,
    *,
    center: bool = True,
) -> dict[str, ConceptVector]:
    """Build concept directions. Centering is on by default -- see ``center_bank``.

    Centering needs at least ~4 concepts to estimate the shared direction; with
    fewer it mostly removes signal. Pass ``center=False`` if you deliberately want
    the raw contrast vectors.
    """
    names = concepts or DEFAULT_CONCEPTS
    bank = {c: build_concept_vector(model, c, layer) for c in names}
    return center_bank(bank) if center and len(bank) >= 4 else bank


def pairwise_cosines(bank: dict[str, ConceptVector]) -> dict[tuple[str, str], float]:
    names = list(bank)
    return {(a, b): cosine(bank[a], bank[b]) for i, a in enumerate(names) for b in names[i + 1 :]}


def max_offdiagonal_cosine(bank: dict[str, ConceptVector]) -> float:
    pairs = pairwise_cosines(bank)
    return max((abs(v) for v in pairs.values()), default=0.0)


def random_control(reference: ConceptVector, *, seed: int = 0) -> ConceptVector:
    """A random direction with the same norm as ``reference``.

    The point of this arm: if the model reports *a* concept just as confidently
    under a meaningless direction, its reports are confabulated rather than read
    off anything real.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(reference.vector.shape, generator=g)
    v = v / v.norm() * reference.vector.norm()
    return ConceptVector(name=f"random[{seed}]", layer=reference.layer, vector=v)


def shuffled_control(reference: ConceptVector, *, seed: int = 0) -> ConceptVector:
    """The reference vector with its coordinates permuted.

    Preserves the norm and the marginal distribution of coordinate magnitudes,
    destroying only the direction. A tighter control than pure Gaussian noise.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(reference.vector.numel(), generator=g)
    return ConceptVector(
        name=f"shuffled[{seed}]", layer=reference.layer, vector=reference.vector[perm].clone()
    )


def cosine(a: ConceptVector, b: ConceptVector) -> float:
    """Sanity check: distinct concepts should not be near-collinear."""
    return float(torch.dot(a.unit(), b.unit()))
