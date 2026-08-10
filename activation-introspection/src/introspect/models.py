"""Model loading and architecture-agnostic access to the residual stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# Instruct models that fit on an M-series laptop in fp16. Introspective report is
# plausibly emergent, so the small models are a plumbing testbed and the scale
# ladder is the actual measurement.
#
# Approximate fp16 weight footprint, before activations and KV cache. On a 24 GB
# machine, stay at or below 7B and load one model at a time (see LoadedModel.free).
KNOWN_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",  # ~1.0 GB
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",  # ~3.1 GB
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",  # ~6.2 GB
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",  # ~15.2 GB -- tight on 24 GB
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

# The scale ladder for the main experiment, smallest first. Whether the
# introspector-observer gap opens up with scale is the question; a single model
# cannot answer it.
SCALE_LADDER = ["qwen-0.5b", "qwen-1.5b", "qwen-3b"]

DEFAULT_MODEL = "qwen-1.5b"

# Rough fp16 weight sizes in GB, for the preflight memory check.
_APPROX_GB = {"qwen-0.5b": 1.0, "qwen-1.5b": 3.1, "qwen-3b": 6.2, "qwen-7b": 15.2}


def memory_warning(name: str) -> str | None:
    """Return a warning if this model is likely to push the machine into swap.

    Swapping on MPS does not fail loudly -- it just makes a sweep take hours
    instead of minutes, which is easy to misread as the model being slow.
    """
    need = _APPROX_GB.get(name)
    if need is None:
        return None
    try:
        import subprocess

        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024**3
    except Exception:
        return None
    # Activations, KV cache, and the OS need headroom well beyond the weights.
    if need > 0.65 * total:
        return f"{name} needs ~{need:.1f} GB of {total:.0f} GB RAM; expect swapping"
    return None


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

    @torch.no_grad()
    def forward_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Full logits for a single forward pass, [batch, seq, vocab]."""
        return cast(torch.Tensor, cast(Any, self.model)(ids).logits)

    def free(self) -> None:
        """Drop the model and reclaim device memory.

        Needed when sweeping several model sizes in one process: MPS does not
        return freed blocks to the OS promptly, and loading 3B on top of a
        still-resident 1.5B is what pushes a 24 GB machine into swap.
        """
        del self.model
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()


def resolve_blocks(model: nn.Module) -> nn.ModuleList:
    """Find the list of transformer blocks across common HF architectures.

    The ``base_model.model.*`` paths cover PEFT/LoRA wrappers, which nest the
    real model one or two levels deeper. Without them, attaching an adapter
    silently breaks every intervention -- the hooks would have nothing to bind
    to, and fine-tuning "on injected examples" would train on clean ones.
    """
    for path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "base_model.model.model.layers",
        "base_model.model.transformer.h",
        "base_model.model.gpt_neox.layers",
    ):
        obj: object = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Could not locate transformer blocks on {type(model).__name__}")


def loaded_revision(model: LoadedModel) -> str:
    """Immutable cache revision attached to the model or tokenizer actually loaded."""
    config = getattr(model.model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    tokenizer_kwargs = getattr(model.tokenizer, "init_kwargs", None)
    if revision is None and isinstance(tokenizer_kwargs, dict):
        revision = tokenizer_kwargs.get("_commit_hash")
    return str(revision) if revision else "unknown"


def load(
    name: str = DEFAULT_MODEL,
    *,
    device: torch.device | None = None,
    revision: str | None = None,
) -> LoadedModel:
    repo = KNOWN_MODELS.get(name, name)
    dev = device or pick_device()
    dtype = pick_dtype(dev)

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(repo, revision=revision),  # type: ignore[no-untyped-call]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, revision=revision),
    )
    model.to(dev)
    cast(Any, model).eval()
    model.requires_grad_(False)

    return LoadedModel(name=repo, model=model, tokenizer=tokenizer, device=dev, dtype=dtype)
