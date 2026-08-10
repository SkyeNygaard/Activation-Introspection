"""Fixed-choice continuation scoring under targeted welfare interventions.

Fast path uses one forward when every option is a single appended token.
Fallback computes sequence log-probability, so Q/K or 0-9 tokenization cannot
silently kill the experiment on a different tokenizer.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
import torch
from torch import Tensor

from welfare_intervention import PublishedActAdd, targeted_published_actadd

def _ctx(blocks, edit):
    return nullcontext() if edit is None else targeted_published_actadd(blocks, edit)

def appended_tokens(tokenizer, prompt: str, continuation: str) -> list[int] | None:
    """Return exact appended token IDs if prompt tokenization is a prefix."""
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + continuation, add_special_tokens=False).input_ids
    if len(full) <= len(base) or full[:len(base)] != base:
        return None
    return [int(x) for x in full[len(base):]]

@torch.no_grad()
def score_continuation(
    model,
    tokenizer,
    blocks,
    prompt: str,
    continuation: str,
    *,
    edit: PublishedActAdd | None = None,
    length_normalize: bool = True,
) -> float:
    base_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full_ids = tokenizer(
        prompt + continuation, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    n_base = int(base_ids.shape[1])
    n_full = int(full_ids.shape[1])
    if n_full <= n_base or not torch.equal(full_ids[:, :n_base], base_ids):
        raise ValueError(
            f"continuation {continuation!r} is not a suffix under tokenizer prefix semantics"
        )

    with _ctx(blocks, edit):
        logits = model(full_ids).logits[0].float()
    # token at position i is predicted by logits i-1.
    target = full_ids[0, n_base:]
    pred = logits[n_base - 1 : n_full - 1]
    lp = torch.log_softmax(pred, dim=-1)
    total = lp.gather(-1, target[:, None]).sum()
    if length_normalize:
        total = total / target.numel()
    return float(total)

@torch.no_grad()
def score_options(
    model,
    tokenizer,
    blocks,
    prompt: str,
    options: Sequence[str],
    *,
    edit: PublishedActAdd | None = None,
    length_normalize: bool = True,
) -> dict[str, object]:
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("need >=2 unique options")

    tokenized = [appended_tokens(tokenizer, prompt, opt) for opt in options]
    if all(ids is not None and len(ids) == 1 for ids in tokenized):
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        with _ctx(blocks, edit):
            logits = model(ids).logits[0, -1].float()
        option_ids = [ids_[0] for ids_ in tokenized]  # type: ignore[index]
        raw = torch.stack([logits[i] for i in option_ids])
        logps = torch.log_softmax(logits, dim=-1)
        scores = [float(logps[i]) for i in option_ids]
        conditional = torch.softmax(raw, dim=0)
        mass = float(torch.logsumexp(raw,0).sub(torch.logsumexp(logits,0)).exp())
        unrestricted_top = int(logits.argmax())
        format_ok = unrestricted_top in set(option_ids)
    else:
        scores = [
            score_continuation(
                model, tokenizer, blocks, prompt, opt,
                edit=edit, length_normalize=length_normalize
            )
            for opt in options
        ]
        conditional = torch.softmax(torch.tensor(scores), dim=0)
        # Multi-token options have no directly comparable single-step label mass.
        mass = float("nan")
        format_ok = False

    best = int(torch.tensor(scores).argmax())
    return {
        "options": list(options),
        "logprobs": scores,
        "conditional_probs": [float(x) for x in conditional],
        "predicted": options[best],
        "single_token_fast_path": all(ids is not None and len(ids) == 1 for ids in tokenized),
        "option_mass": mass,
        "format_ok": format_ok,
    }
