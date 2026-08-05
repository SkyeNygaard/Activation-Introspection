"""Retained-trace endpoint: can a removed edit still be used after the fact?

The schedule is the point. Every earlier design in this repo scored an answer
while the intervention was still live, which lets the edit promote the answer
token directly (see the token-promotion control in ``notes/03-lab-notebook.md``).
Here the edit exists only while a neutral carrier's KV cache is being built:

    stage 1   chat prefix + carrier forward, hook live, cache retained
    ---       hook removed, asserted absent
    stage 2   a freshly sampled concept->label codebook is appended, then the
              query; the label is scored

Because the codebook is sampled *after* the cache exists, the injected vector
cannot have targeted whichever label happens to be correct. Chance is exactly
``1 / width`` for any arm carrying no concept information.

Note what the ``clean`` and ``sham`` arms can and cannot show. They run one
forward per (carrier, codebook) and score it against every concept; the codebooks
are cyclic, so exactly one of the eight scored rows is correct and their accuracy
is ``1 / width`` by arithmetic no matter what the model does. That makes them
pipeline diagnostics -- the hook plumbing ran, no label leaked through the prompt
itself -- and not evidence about the concept. The arms that can actually fail are
``random`` and ``shuffled``, which carry a per-concept edit and could come out
above chance if the effect were an artifact of perturbation rather than identity.

The same stage-1 forward also captures the residual stream at the last carrier
position at every requested layer, so decodability and usability are measured on
the *identical* retained state. That site matching is what the retracted
``r = -0.774`` comparison lacked.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from transformers import DynamicCache

from introspect.concepts import ConceptVector
from introspect.hooks import Intervention, capture, intervene
from introspect.models import LoadedModel

# Labels are semantically inert and, on Qwen, single tokens after a space. A
# label must never be a word the injected direction could promote lexically.
LABEL_ALPHABET = ["Q", "K", "Z", "J", "X", "V", "W", "Y"]

# Two disjoint concept banks, matched for concreteness. Layer, strength, and any
# prompt repair are chosen on DEV only; TEST is run once, after those choices are
# frozen. Reusing one bank for both is how a tuned pipeline turns noise into a
# headline.
DEV_CONCEPTS = ["ocean", "bread", "volcano", "violin", "spider", "hospital", "desert", "clock"]
TEST_CONCEPTS = ["mountain", "coffee", "tiger", "piano", "forest", "airport", "candle", "river"]

# Neutral carriers. They must never name a concept, and they are identical
# across arms: the only thing that differs between arms is the hidden edit.
CARRIER_FAMILIES = [
    "Internal note, filed without a heading. The quarterly maintenance window "
    "opens on Tuesday and the archive index was rebuilt overnight.",
    "Transcript fragment, speaker unlabelled. The room was cold, the projector "
    "had been left running, and nobody had reset the counter on the door.",
    "Log excerpt. Rotation completed, checksum verified, three entries deferred "
    "to the next cycle without comment from the operator on duty.",
]

QUERY = (
    "One of the concepts listed in the codebook above has been activated in your "
    "context by a direct edit to your internal state. Using the codebook, reply "
    "with that concept's single label character and nothing else."
)

# The ceiling arm. The concept is stated in ordinary text instead of injected;
# everything downstream -- carrier length, codebook, query, scoring -- is
# identical. Without this, a null in the injected arms cannot be distinguished
# from the model simply being unable to do delayed codebook binding at all.
NATURAL_PREFIX = "The subject of this note is {concept}. "

ANSWER_PREFIX = "Label:"

# Position 0 carries the attention sink, whose magnitude is orders of magnitude
# above a normal residual. Editing there swamps the concept direction.
SINK_SKIP = 4


def natural_carrier(carrier: str, concept: str) -> str:
    """Carrier for the ceiling arm, with the concept stated in plain text."""
    return NATURAL_PREFIX.format(concept=concept) + carrier


@dataclass(frozen=True)
class Codebook:
    """A concept -> label assignment sampled after the carrier cache exists.

    ``order`` is the sequence the lines are *displayed* in, which is deliberately
    decoupled from the label assignment: otherwise a model that simply favours
    one line of the codebook would favour one label, and per-concept accuracy
    would inherit a position artifact.
    """

    mapping: dict[str, str]
    order: tuple[str, ...]

    @property
    def labels(self) -> list[str]:
        return [self.mapping[c] for c in self.order]

    def render(self) -> str:
        lines = "\n".join(f"  {c} = {self.mapping[c]}" for c in self.order)
        return f"Codebook (assigned just now, use only this):\n{lines}"

    def digest(self) -> str:
        payload = ";".join(f"{c}={self.mapping[c]}" for c in self.order)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def balanced_codebooks(
    concepts: Sequence[str], labels: Sequence[str], *, seed: int = 0
) -> list[Codebook]:
    """Cyclic label shifts, with display order permuted independently.

    The cyclic shift is what makes chance exactly ``1/n``: each concept occupies
    each label slot exactly once across the returned books. A random draw per
    trial would leave slot frequency unbalanced at these sample sizes, and a slot
    bias would then be indistinguishable from a real effect.

    The display order is permuted by a separate seeded stream so that "the model
    likes line 1" and "the model likes label Q" cannot be the same artifact.
    """
    n = len(concepts)
    if len(labels) < n:
        raise ValueError(f"need {n} labels for {n} concepts, got {len(labels)}")
    rng = torch.Generator().manual_seed(seed)
    books = []
    for s in range(n):
        mapping = {c: labels[(i + s) % n] for i, c in enumerate(concepts)}
        perm = torch.randperm(n, generator=rng).tolist()
        books.append(Codebook(mapping=mapping, order=tuple(concepts[i] for i in perm)))
    return books


def single_token_labels(model: LoadedModel, alphabet: Sequence[str] = LABEL_ALPHABET) -> list[str]:
    """Labels that tokenize to exactly one id after a leading space.

    Multi-token labels would make the scored position ambiguous and silently
    change what the probability means.
    """
    out: list[str] = []
    for ch in alphabet:
        ids = model.tokenizer(" " + ch, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append(ch)
    return out


def label_token_ids(model: LoadedModel, labels: Sequence[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for lab in labels:
        tok = model.tokenizer(" " + lab, add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f"label {lab!r} is not a single token")
        ids[lab] = int(tok[0])
    return ids


@dataclass(frozen=True)
class Split:
    """Token ids for the two stages, with the split point verified."""

    stage1: Tensor
    stage2: Tensor
    n_carrier: int


def split_prompt(model: LoadedModel, carrier: str, codebook: Codebook) -> Split:
    """Tokenize the full prompt once, then cut it at the carrier boundary.

    Tokenizing the halves separately would be wrong: BPE merges across a
    boundary, so ``encode(a) + encode(b) != encode(a + b)`` in general, and the
    two stages would not reconstruct the sequence the model is supposed to see.

    So the full prompt is tokenized once and sliced at the largest ``k`` whose
    decoding is still a prefix of the carrier text. That guarantees the property
    the design actually needs -- **no codebook token is ever in stage 1** -- for
    any carrier text, without assuming a stable merge at the seam. Any carrier
    tokens past ``k`` simply run in stage 2, unedited, which is harmless: they
    are identical across arms.
    """
    head = model.chat(carrier, assistant_prefix="")
    prefix_text = head[: head.index(carrier) + len(carrier)]
    tail_text = f"\n\n{codebook.render()}\n\n{QUERY}"
    full_text = model.chat(carrier + tail_text, assistant_prefix=ANSWER_PREFIX)
    full = model.tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids

    def decodes_within_carrier(k: int) -> bool:
        text = model.tokenizer.decode(full[0, :k], skip_special_tokens=False)
        return prefix_text.startswith(text)

    lo, hi = 0, int(full.shape[1])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decodes_within_carrier(mid):
            lo = mid
        else:
            hi = mid - 1
    if lo <= SINK_SKIP:
        raise ValueError(f"carrier boundary landed at token {lo}; carrier is too short")
    return Split(
        stage1=full[:, :lo].to(model.device),
        stage2=full[:, lo:].to(model.device),
        n_carrier=lo,
    )


@dataclass
class TrialResult:
    """One (carrier, codebook, arm) observation plus its validity diagnostics."""

    label_probs: dict[str, float]
    pred_label: str
    # True when the unrestricted argmax over the whole vocabulary is a label at
    # all. A collapse here means the arm broke the model's formatting, and its
    # label probabilities are not comparable with an intact arm.
    format_ok: bool
    carrier_kl: float
    carrier_logits: Tensor
    # residual at the last carrier position, keyed by layer, from the SAME
    # stage-1 forward that produced the answer.
    carrier_acts: dict[int, Tensor] = field(default_factory=dict)


def _assert_no_hooks(model: LoadedModel) -> None:
    live = sum(len(b._forward_hooks) for b in model.blocks)
    if live:
        raise RuntimeError(f"{live} forward hook(s) still registered at stage 2")


@dataclass
class CarrierCache:
    """A retained carrier state, reusable across codebooks.

    The carrier forward does not depend on the codebook -- that is the whole
    point of the schedule -- so building it once and replaying stage 2 against a
    cropped copy is both faster and a stronger guarantee that every codebook
    sees a byte-identical retained state.
    """

    cache: DynamicCache
    n_carrier: int
    carrier_logits: Tensor
    acts: dict[int, Tensor]


@torch.no_grad()
def build_carrier_cache(
    model: LoadedModel,
    stage1_ids: Tensor,
    *,
    interventions: Sequence[Intervention] = (),
    capture_layers: Sequence[int] = (),
) -> CarrierCache:
    """Stage 1: forward the carrier under ``interventions`` and keep the cache."""
    cache = DynamicCache()
    with intervene(model, interventions), capture(model, capture_layers) as store:
        out = model.model(stage1_ids, past_key_values=cache, use_cache=True)

    # The hook must be gone before the codebook enters the forward pass at all.
    _assert_no_hooks(model)

    acts = {
        layer: store.acts[layer][0][:, -1, :].squeeze(0).clone()
        for layer in capture_layers
        if layer in store.acts
    }
    return CarrierCache(
        cache=cache,
        n_carrier=int(stage1_ids.shape[1]),
        carrier_logits=out.logits[0, -1, :].float().cpu(),
        acts=acts,
    )


@torch.no_grad()
def query_cache(
    model: LoadedModel,
    carrier: CarrierCache,
    stage2_ids: Tensor,
    codebook: Codebook,
    label_ids: dict[str, int],
) -> tuple[dict[str, float], str, bool]:
    """Stage 2: append codebook + query to the retained cache and score labels.

    Crops the cache back to the carrier afterwards so the same retained state
    can be replayed against the next codebook.
    """
    _assert_no_hooks(model)
    try:
        out = model.model(stage2_ids, past_key_values=carrier.cache, use_cache=True)
        logits = out.logits[0, -1, :].float()
    finally:
        carrier.cache.crop(carrier.n_carrier)

    cand = torch.tensor([label_ids[lab] for lab in codebook.labels], device=logits.device)
    sub = torch.softmax(logits[cand], dim=-1)
    probs = {lab: float(sub[i]) for i, lab in enumerate(codebook.labels)}
    pred = max(probs, key=lambda k: probs[k])
    format_ok = int(logits.argmax()) in set(label_ids.values())
    return probs, pred, format_ok


def carrier_kl(clean_logits: Tensor, arm_logits: Tensor) -> float:
    """KL(clean || arm) at the last carrier position: how much the edit damaged.

    Arms are only comparable inside a matched KL band; an arm that mangles the
    carrier is not a control for one that does not.
    """
    p = torch.log_softmax(clean_logits, dim=-1)
    q = torch.log_softmax(arm_logits, dim=-1)
    return float(torch.sum(p.exp() * (p - q)))


@torch.no_grad()
def run_trial(
    model: LoadedModel,
    split: Split,
    codebook: Codebook,
    *,
    interventions: Sequence[Intervention] = (),
    capture_layers: Sequence[int] = (),
    label_ids: dict[str, int],
    clean_carrier_logits: Tensor | None = None,
) -> TrialResult:
    """Single-trial convenience wrapper over the two stages."""
    carrier = build_carrier_cache(
        model, split.stage1, interventions=interventions, capture_layers=capture_layers
    )
    probs, pred, format_ok = query_cache(model, carrier, split.stage2, codebook, label_ids)
    kl = (
        0.0
        if clean_carrier_logits is None
        else carrier_kl(clean_carrier_logits, carrier.carrier_logits)
    )
    return TrialResult(
        label_probs=probs,
        pred_label=pred,
        format_ok=format_ok,
        carrier_kl=kl,
        carrier_logits=carrier.carrier_logits,
        carrier_acts=carrier.acts,
    )


def carrier_positions(n_carrier: int) -> list[int]:
    """Carrier token indices to edit, excluding the attention-sink region."""
    if n_carrier <= SINK_SKIP:
        raise ValueError(f"carrier is only {n_carrier} tokens; need more than {SINK_SKIP}")
    return list(range(SINK_SKIP, n_carrier))


def build_arm(
    arm: str,
    vector: ConceptVector,
    layer: int,
    strength: float,
    positions: list[int],
) -> list[Intervention]:
    """Interventions for one arm.

    ``sham`` runs the identical hook with strength zero, so hook plumbing and
    any cost of merely running a hook are held constant. ``clean`` registers no
    hook at all; the pair separates "an edit happened" from "a hook ran".
    """
    if arm == "clean":
        return []
    effective = 0.0 if arm == "sham" else strength
    return [
        Intervention(
            layer=layer,
            direction=vector.vector,
            strength=effective,
            positions=positions,
            label=f"{arm}:{vector.name}",
        )
    ]
