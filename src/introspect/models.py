"""Model loading and architecture-agnostic access to the residual stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# Small instruct models that fit comfortably on an M-series laptop in fp16/bf16.
# Introspective report is plausibly an emergent capability, so treat the small
# models as a plumbing testbed and the larger ones as the real measurement.
KNOWN_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

DEFAULT_MODEL = "qwen-1.5b"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # MPS supports fp16 well; bf16 support is patchy across torch versions.
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


@dataclass
class LoadedModel:
    """A causal LM plus the handles needed to read and write its residual stream."""

    name: str
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype

    @property
    def blocks(self) -> nn.ModuleList:
        """The transformer block list, whatever the architecture calls it."""
        return resolve_blocks(self.model)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def d_model(self) -> int:
        # nn.Module.__getattr__ is typed as Tensor | Module, so every attribute
        # reached through it needs a cast. This is a limitation of the torch stubs,
        # not a real ambiguity.
        return int(cast(Any, self.model).config.hidden_size)

    def chat(self, user: str, assistant_prefix: str = "") -> str:
        """Render a single-turn chat prompt, optionally prefilling the reply."""
        messages = [{"role": "user", "content": user}]
        text = cast(
            str,
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
        return text + assistant_prefix

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return cast(torch.Tensor, ids).to(self.device)

    def generate_ids(self, ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Wrap ``model.generate``, which the stubs do not expose on nn.Module."""
        return cast(torch.Tensor, cast(Any, self.model).generate(ids, **kwargs))


def resolve_blocks(model: nn.Module) -> nn.ModuleList:
    """Find the list of transformer blocks across common HF architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj: object = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Could not locate transformer blocks on {type(model).__name__}")


def load(name: str = DEFAULT_MODEL, *, device: torch.device | None = None) -> LoadedModel:
    repo = KNOWN_MODELS.get(name, name)
    dev = device or pick_device()
    dtype = pick_dtype(dev)

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(repo),  # type: ignore[no-untyped-call]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cast(nn.Module, AutoModelForCausalLM.from_pretrained(repo, dtype=dtype))
    model.to(dev)
    cast(Any, model).eval()
    model.requires_grad_(False)

    return LoadedModel(name=repo, model=model, tokenizer=tokenizer, device=dev, dtype=dtype)
