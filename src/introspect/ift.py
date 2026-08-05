"""Introspection fine-tuning, and a corrected layer-generalization pilot.

## The question

*Introspection Fine-Tuning* (arXiv 2607.14111) shows that supervised fine-tuning
on a model's own perturbed forward passes teaches small models to report
perturbations — Llama-1B goes from 9.6% to 60.6% on sentence localization. The
local task is concept identification, not sentence localization, so this is not
a reproduction of IFT.

An earlier version compared a fixed-source propagation profile (inject at L8,
read at L) to this module's held-out-layer evaluation (inject at L, read at the
output). The resulting negative correlation was an estimand mismatch and is
retracted. Only an inject-at-L/read-at-output profile is site-matched.

## Why the digit format is the right target here

Digit-indexed forced choice taxes an untrained small model heavily (measured
elsewhere in this repo: free-form 0.33 where digits scored 0.05). That makes the
*pre*-training number hard to interpret. Randomizing the option mapping removes
the stable concept→digit shortcut, and digit tokens avoid the direct lexical
token-promotion confound. Format competence, held-out concepts, sham vectors,
and independent training runs are still required before the endpoint supports
an introspection claim.

## Training signal

Cross-entropy on the correct digit token at the final prompt position, with the
concept injected during the forward pass via the same hooks used everywhere else.
Only LoRA parameters are updated. Base weights remain frozen, but an active
adapter changes forward activations, so post-training representations are not
assumed unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from introspect.concepts import ConceptVector
from introspect.grading import score_choices
from introspect.hooks import Intervention, intervene
from introspect.models import LoadedModel
from introspect.prompts import (
    IDENTIFY_FORCED_CHOICE_VARIANTS,
    forced_choice,
    permuted_options,
    variant,
)

# Paraphrase indices reserved for training vs evaluation. There are 5 variants;
# holding two back means post-training accuracy is measured on wordings the
# adapter never saw, not only on layers it never saw.
TRAIN_VARIANTS = (0, 1, 2)
EVAL_VARIANTS = (3, 4)


def seeds_for(variants: Sequence[int], n_per_variant: int) -> list[int]:
    """Trial ids whose ``variant()`` index falls in ``variants``.

    Repeated ids within a paraphrase select different option permutations; they
    are not independent model seeds and must not be treated as such in error
    bars.  The independent units in a confirmatory run are model-training seeds,
    held-out concepts, and genuinely distinct prompt families.
    """
    n_v = len(IDENTIFY_FORCED_CHOICE_VARIANTS)
    return [v + n_v * k for k in range(n_per_variant) for v in variants]


@dataclass
class Example:
    prompt: str
    concept_name: str
    concept_idx: int
    layer: int
    strength: float
    vector: Tensor


def _validate_bank_layer(bank: dict[str, ConceptVector], layer: int) -> None:
    """Refuse to inject vectors constructed at a different residual site."""
    wrong = sorted({cv.layer for cv in bank.values()} - {layer})
    if wrong:
        raise ValueError(
            f"layer {layer} requires a bank constructed at layer {layer}; "
            f"received vector layer(s) {wrong}"
        )


def build_examples(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    layers: Sequence[int],
    strengths: Sequence[float],
    *,
    seeds: Sequence[int],
) -> list[Example]:
    """One example per (concept, layer, strength, seed).

    ``seeds`` selects prompt paraphrases via ``variant``. Training and evaluation
    are given disjoint seed sets whose paraphrases do not overlap, so the model is
    scored on wordings it never saw as well as on layers it never saw.
    """
    concepts = sorted(bank)
    out: list[Example] = []
    for name in concepts:
        for layer in layers:
            _validate_bank_layer(bank, layer)
            for strength in strengths:
                for seed in seeds:
                    options = permuted_options(concepts, seed)
                    target_idx = options.index(name)
                    prompt = model.chat(
                        variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(
                            options=forced_choice(options)
                        )
                    )
                    out.append(
                        Example(
                            prompt=prompt,
                            concept_name=name,
                            concept_idx=target_idx,
                            layer=layer,
                            strength=strength,
                            vector=bank[name].vector,
                        )
                    )
    return out


def digit_token_ids(model: LoadedModel, n: int) -> list[int]:
    ids = []
    for i in range(n):
        toks = model.tokenizer(str(i + 1), add_special_tokens=False).input_ids
        assert len(toks) == 1, f"digit {i + 1} is not a single token"
        ids.append(int(toks[0]))
    return ids


def attach_lora(model: LoadedModel, *, r: int = 16, alpha: int = 32) -> object:
    """Wrap the base model in a LoRA adapter, leaving base weights frozen.

    Targets attention and MLP projections. Base weights remain frozen, but the
    active adapter changes the forward activations; it is therefore incorrect to
    describe post-IFT representations as unchanged.  The wrapper is returned in
    evaluation mode so LoRA dropout cannot leak into baseline evaluation.
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_model = get_peft_model(cast(Any, model.model), cfg)
    peft_model.eval()
    model.model = peft_model
    return peft_model


def train(
    model: LoadedModel,
    examples: list[Example],
    digit_ids: list[int],
    *,
    epochs: int = 2,
    lr: float = 2e-4,
    seed: int = 0,
    log_every: int = 100,
) -> list[float]:
    """Fine-tune on injected examples. Returns the loss curve."""
    rng = random.Random(seed)
    params = [p for p in model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    digit_tensor = torch.tensor(digit_ids, device=model.device)

    losses: list[float] = []
    was_training = model.model.training
    model.model.train()
    try:
        for epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for step, i in enumerate(order):
                ex = examples[i]
                ids = model.encode(ex.prompt)
                iv = Intervention(layer=ex.layer, direction=ex.vector, strength=ex.strength)

                with intervene(model, [iv], prompt_len=int(ids.shape[1])):
                    logits = model.model(ids).logits

                # Restrict the loss to the digit options: the task is choosing among
                # them, not predicting arbitrary vocabulary.
                option_logits = logits[0, -1, digit_tensor].float().unsqueeze(0)
                target = torch.tensor([ex.concept_idx], device=model.device)
                loss = torch.nn.functional.cross_entropy(option_logits, target)

                cast(Any, loss).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))

                if log_every and (epoch * len(order) + step) % log_every == 0:
                    recent = sum(losses[-log_every:]) / len(losses[-log_every:])
                    print(
                        f"  epoch {epoch} step {step}/{len(order)}  loss {recent:.4f}",
                        flush=True,
                    )
    finally:
        model.model.train(was_training)
    return losses


@torch.no_grad()
def evaluate_layer(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    layer: int,
    strength: float,
    *,
    seeds: Sequence[int],
) -> list[bool]:
    """Identification accuracy at one layer, on held-out prompt paraphrases.

    The vector bank must have been constructed at ``layer``. Evaluation is
    always performed with dropout disabled and restores the caller's prior mode.
    """
    _validate_bank_layer(bank, layer)
    concepts = sorted(bank)
    digits = [str(i + 1) for i in range(len(concepts))]
    correct: list[bool] = []
    was_training = model.model.training
    model.model.eval()
    try:
        for name in concepts:
            iv = Intervention(layer=layer, direction=bank[name].vector, strength=strength)
            for seed in seeds:
                options = permuted_options(concepts, seed)
                prompt = model.chat(
                    variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(
                        options=forced_choice(options)
                    )
                )
                choice = score_choices(model, prompt, digits, interventions=[iv])
                correct.append(choice.argmax == options.index(name))
    finally:
        model.model.train(was_training)
    return correct
