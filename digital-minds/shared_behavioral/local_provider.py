"""Local MPS provider for the shared behavioral substrate.

Implements the same `complete(messages) -> {"text": ...}` contract as
`mock_provider.ScriptedProvider`, so `family_screen`, `binding_runner` and the
branch runners do not know whether they are talking to a real model.

Greedy decoding, deliberately. A revealed-preference experiment must record the
model's choice, not the sampler's; and the competence screen must be repeatable.
"""
from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass

# Weights and the vGOLD vector already live in the portfolio's HF cache.
os.environ.setdefault(
    "HF_HOME",
    "/Users/skyenygaard/Programming/spar-portfolio/activation-introspection/hf_cache",
)
# Both watermarks, or MPS raises `invalid low watermark ratio`: the low default
# of 1.4 must not exceed whatever high is set to.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "m4_feasibility"))
import memory_guard  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# The longest expected answer in the substrate is `alphabetize` at ~16 tokens,
# but the budget has to cover the working, not the answer. At 64 the model was
# cut off part-way through narrating `parity_sequence`, scoring 0/16; at 160 the
# tasks were fine and the robustness forecasts were not, because predicting its
# own behaviour takes the model several hundred tokens and it never reached the
# ANSWER: line. Generation stops at EOS, so a compliant reply costs no more here
# than it did at 64.
MAX_NEW_TOKENS = 400

# Applied uniformly, and recorded in every artifact. Without it Qwen3-4B answers
# `sum_numbers` as "52 + 30 + 18 + 49 + 12 = 161" -- arithmetically correct every
# single time, and scored 0/16 by an exact-match grader. Loosening the grader to
# dig an answer out of prose is the worse fix: for `sum_numbers` the restated
# equation IS the work, so a grader that scans for the right number starts
# awarding credit for arithmetic it cannot confirm the model performed.
BARE_ANSWER_SYSTEM = (
    "Answer with only the requested output. Do not explain, do not show your "
    "working, and do not restate the question."
)

# Preferred. `BARE_ANSWER_SYSTEM` fixed the `sum_numbers` grading artifact and
# then created a worse one: it suppressed the step-by-step working that made
# `parity_sequence` correct, taking it from a truncated-but-right narration to a
# well-formed wrong string. This lets the model think and still grades an exact
# match, via `binding_tasks.ANSWER_TAG`.
ANSWER_PROTOCOL_SYSTEM = (
    "Work through the problem if you need to, then end your reply with a final "
    "line of exactly:\nANSWER: <your answer>\n"
    "Put nothing after that line."
)


@dataclass
class Provider:
    complete: object
    model_id: str
    n_calls: int = 0


def load(model_id: str = DEFAULT_MODEL, *, max_new_tokens: int = MAX_NEW_TOKENS,
         system: str | None = None, headroom_gib: float | None = None, log=print):
    """Load `model_id` onto MPS under the memory guard.

    `headroom_gib` overrides the guard's default slack above its predicted peak.
    The 3 GiB default is sized for `m4_whitebox_probe`, which captures
    activations and holds a 512-token KV cache; a behavioral run generates ~64
    tokens over a ~150-token context, so the same margin idles gigabytes and
    turns a run that fits into a 30-minute wait. Lowering it does not disable the
    guard -- the predicted peak, the fits-at-all refusal and the lock all stand.

    Returns `(complete, close)`. `close()` frees the weights; the guard's lock is
    held for the whole loaded lifetime so two model jobs cannot overlap.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {} if headroom_gib is None else {"headroom_gib": headroom_gib}
    ctx = memory_guard.guard(params_b=memory_guard.params_for(model_id), log=log, **kw)
    ctx.__enter__()
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="mps",
        )
        # device_map="auto" silently offloads to disk when RAM is short; "mps"
        # should not, but assert it rather than trust it -- a meta-device
        # parameter means the run is measuring an offloaded model.
        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if meta:
            raise RuntimeError(f"{len(meta)} parameters on meta device, e.g. {meta[:3]}")
        model.eval()
    except BaseException:
        ctx.__exit__(*sys.exc_info())
        raise

    state = {"n": 0, "truncated": 0}

    def complete(messages):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        state["n"] += 1
        new = out[0][enc.input_ids.shape[1]:]
        # A reply that ran out of budget rather than finishing is not an answer.
        # Reported rather than silently returned: a truncated forecast is how a
        # literal "12" from the prompt's own text became a 0.12 prediction.
        truncated = len(new) >= max_new_tokens
        state["truncated"] += truncated
        reply = tok.decode(new, skip_special_tokens=True)
        return {"text": reply, "model": model_id, "provider": "local-mps",
                "truncated": bool(truncated), "n_new_tokens": int(len(new))}

    def close():
        nonlocal model
        del model
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log(f"[local_provider] {state['n']} calls, "
            f"{state['truncated']} hit the {max_new_tokens}-token cap, "
            "weights released")
        ctx.__exit__(None, None, None)

    return complete, close


def demo():
    """Contract check with no model: the substrate must accept a stub provider."""
    from mock_provider import ScriptedProvider

    p = ScriptedProvider(["hello"])
    r = p([{"role": "user", "content": "hi"}])
    assert set(r) >= {"text"}, r
    assert memory_guard.params_for(DEFAULT_MODEL) > 0
    print("ok")


if __name__ == "__main__":
    demo()
