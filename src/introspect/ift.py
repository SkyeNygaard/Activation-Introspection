"""Introspection fine-tuning, and the layer-generalization test it enables.

## The question

*Introspection Fine-Tuning* (arXiv 2607.14111) shows that supervised fine-tuning
on a model's own perturbed forward passes teaches small models to report
perturbations — Llama-1B goes from 9.6% to 60.6% on sentence localization. It
reports peak accuracy "at optimal layer/strength configurations" and lists
**"mechanisms underlying the layer-agnostic generalization effect"** as an open
question.

`scripts/layer_profile.py` measures a candidate mechanism: how much concept
identity is linearly decodable, before any training, from the state the model
answers from. That yields a prediction this module tests directly:

    Train the model to report injections at ONE layer. Its accuracy at
    held-out layers should track the pre-training transfer-probe profile —
    high where the injected state already resembles the model's own
    representation of the concept, at chance where it does not.

If it holds, the profile predicts where introspection training will generalize
*without running the training*, which is the useful form of the result.

## Why the digit format is the right target here

Digit-indexed forced choice taxes an untrained small model heavily (measured
elsewhere in this repo: free-form 0.33 where digits scored 0.05). That makes the
*pre*-training number uninterpretable — but it is exactly what fine-tuning
removes. Post-training digit accuracy is therefore a clean measure of the thing
we care about, and it is not circular: digit tokens have no lexical overlap with
the injected concept, so the injection cannot promote the answer.

## Training signal

Cross-entropy on the correct digit token at the final prompt position, with the
concept injected during the forward pass via the same hooks used everywhere else.
Only LoRA parameters are updated; the base model is frozen, so the intervention
machinery and the model's representations are unchanged.
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
from introspect.prompts import IDENTIFY_FORCED_CHOICE_VARIANTS, forced_choice, variant

# Paraphrase indices reserved for training vs evaluation. There are 5 variants;
# holding two back means post-training accuracy is measured on wordings the
# adapter never saw, not only on layers it never saw.
TRAIN_VARIANTS = (0, 1, 2)
EVAL_VARIANTS = (3, 4)


def seeds_for(variants: Sequence[int], n_per_variant: int) -> list[int]:
    """Seeds whose ``variant()`` index falls in ``variants``."""
    n_v = len(IDENTIFY_FORCED_CHOICE_VARIANTS)
    return [v + n_v * k for k in range(n_per_variant) for v in variants]


@dataclass
class Example:
    prompt: str
    concept_idx: int
    layer: int
    strength: float
    vector: Tensor


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
    option_block = forced_choice(concepts)
    out: list[Example] = []
    for idx, name in enumerate(concepts):
        for layer in layers:
            for strength in strengths:
                for seed in seeds:
                    prompt = model.chat(
                        variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(options=option_block)
                    )
                    out.append(
                        Example(
                            prompt=prompt,
                            concept_idx=idx,
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

    Targets attention and MLP projections. The base model is frozen so the
    representations the transfer probe measured are unchanged -- fine-tuning
    changes how the model *answers*, not what it internally represents, which is
    what makes the layer-generalization comparison meaningful.
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
                print(f"  epoch {epoch} step {step}/{len(order)}  loss {recent:.4f}", flush=True)
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
    """Identification accuracy at one layer, on held-out prompt paraphrases."""
    concepts = sorted(bank)
    option_block = forced_choice(concepts)
    digits = [str(i + 1) for i in range(len(concepts))]
    correct: list[bool] = []
    for idx, name in enumerate(concepts):
        iv = Intervention(layer=layer, direction=bank[name].vector, strength=strength)
        for seed in seeds:
            prompt = model.chat(
                variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(options=option_block)
            )
            choice = score_choices(model, prompt, digits, interventions=[iv])
            correct.append(choice.argmax == idx)
    return correct
