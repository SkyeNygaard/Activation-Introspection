"""The fair-at-higher-cost comparison.

The headline comparison gives both sides one forward pass on the question alone.
That is the cost an auditor is trying to save. But the damage is measured over a
whole answer, so a single-token output measure is being asked to predict a
hundred-token outcome, and a critic would rightly say the outside signal was
handicapped.

So: replay the untouched model's own answers through both versions and accumulate
the divergence across the answer. Both sides get the same longer look --
  outside -> the next-word disagreement averaged over the answer
  inside  -> the internal difference averaged over the answer
This costs a generation, which is the thing the ranking was supposed to avoid, so
it is a control rather than a rival method.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--per-question", type=int, default=2)
    args = ap.parse_args()

    gens = json.load(open(f"results/{args.tag}_generations.json"))
    by_q: dict[str, list[str]] = {}
    for g in gens:
        if g["version"] == "base":
            by_q.setdefault(g["id"], []).append(g["answer"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out = {}
    for n, (qid, answers) in enumerate(by_q.items()):
        kls, norms = [], []
        for ans in answers[: args.per_question]:
            q = next(g["question"] for g in gens if g["id"] == qid)
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            p_ids = tok(prompt, return_tensors="pt").input_ids
            full = tok(prompt + ans, return_tensors="pt").input_ids.to(device)
            start = p_ids.shape[1]
            if full.shape[1] <= start + 1:
                continue
            with torch.no_grad():
                t = model(full, output_hidden_states=True)
                with model.disable_adapter():
                    b = model(full, output_hidden_states=True)
            # Positions that predict the answer tokens.
            lp = torch.log_softmax(b.logits[0, start - 1 : -1].float(), -1)
            lq = torch.log_softmax(t.logits[0, start - 1 : -1].float(), -1)
            kls.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
            d = torch.stack(
                [(x[0, start:] - y[0, start:]).float().norm(dim=-1).mean()
                 for y, x in zip(b.hidden_states, t.hidden_states)]
            )
            norms.append(d.cpu().numpy())
            del b, t
        if kls:
            out[qid] = {"kl_forced": float(np.mean(kls)),
                        "norm_forced": np.mean(norms, axis=0).tolist()}
        if n % 25 == 0:
            print(f"  {n+1}/{len(by_q)}", flush=True)

    json.dump(out, open(f"results/{args.tag}_forced.json", "w"))
    print(f"wrote results/{args.tag}_forced.json ({len(out)} questions)")


if __name__ == "__main__":
    main()
