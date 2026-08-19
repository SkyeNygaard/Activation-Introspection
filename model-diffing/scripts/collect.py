"""Collect, for every question: the cheap signals, and the answers that give ground truth.

One copy of the model is loaded; the LoRA adapter is toggled to switch between the
"before" and "after" versions, so the pair costs the memory of one model.

Signals are computed from the question alone, with nothing generated. Answers are
generated separately and judged elsewhere -- this script never decides what counts
as a behaviour change.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TOPK = 256  # tokens of each model's next-word distribution an API auditor could see


def load_pool(limit_per_topic: int, where: str = "data/questions") -> list[dict]:
    out = []
    for path in sorted(glob.glob(f"{where}/*.yaml")):
        topic = os.path.basename(path).removesuffix(".yaml")
        for item in yaml.safe_load(open(path))[:limit_per_topic]:
            out.append({"id": item["id"], "topic": topic, "question": item["question"]})
    return out


def outside_features(base_logits: torch.Tensor, tuned_logits: torch.Tensor) -> dict[str, float]:
    """Everything an auditor with next-word probabilities could compute. No internals."""
    lp, lq = torch.log_softmax(base_logits, -1), torch.log_softmax(tuned_logits, -1)
    p, q = lp.exp(), lq.exp()
    kl = float((p * (lp - lq)).sum())
    rkl = float((q * (lq - lp)).sum())
    tv = float(0.5 * (p - q).abs().sum())
    top_base = int(base_logits.argmax())
    pb, _ = lp.topk(TOPK)
    qb, _ = lq.topk(TOPK)
    return {
        "kl": kl,
        "reverse_kl": rkl,
        "jsd": 0.5 * (kl + rkl),
        "tv": tv,
        "entropy_base": float(-(p * lp).sum()),
        "entropy_tuned": float(-(q * lq).sum()),
        "top1_agree": float(top_base == int(tuned_logits.argmax())),
        "max_logprob_shift": float((lp - lq).abs().max()),
        "top1_prob_base": float(p.max()),
        "top1_prob_tuned": float(q.max()),
        "topk_mass_base": float(pb.exp().sum()),
        "topk_mass_tuned": float(qb.exp().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice")
    ap.add_argument("--tag", default="medical7b")
    ap.add_argument("--per-topic", type=int, default=30)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signals-only", action="store_true")
    ap.add_argument("--questions", default="data/questions")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    pool = load_pool(args.per_topic, args.questions)
    print(f"{len(pool)} questions, {args.samples} answers per version, seed {args.seed}", flush=True)

    deltas, rows, gens = [], [], []
    t0 = time.time()
    for n, item in enumerate(pool):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            tuned = model(**ids, output_hidden_states=True)
            with model.disable_adapter():
                base = model(**ids, output_hidden_states=True)
        # Last question token: everything the model has worked out before it speaks.
        d = torch.stack(
            [(t[0, -1] - b[0, -1]).float() for b, t in zip(base.hidden_states, tuned.hidden_states)]
        )
        deltas.append(d.cpu().numpy())
        rows.append(
            {**item, **outside_features(base.logits[0, -1].float(), tuned.logits[0, -1].float())}
        )
        del base, tuned

        if not args.signals_only:
            for label, off in (("base", True), ("tuned", False)):
                torch.manual_seed(args.seed + n)
                with torch.no_grad():
                    ctx = model.disable_adapter() if off else _null()
                    with ctx:
                        out = model.generate(
                            **ids,
                            max_new_tokens=args.max_new,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            num_return_sequences=args.samples,
                            pad_token_id=tok.eos_token_id,
                        )
                for k, o in enumerate(out):
                    gens.append(
                        {
                            "id": item["id"],
                            "topic": item["topic"],
                            "question": item["question"],
                            "version": label,
                            "sample": k,
                            "answer": tok.decode(
                                o[ids["input_ids"].shape[1] :], skip_special_tokens=True
                            ).strip(),
                        }
                    )
        if n % 10 == 0:
            rate = (time.time() - t0) / (n + 1)
            print(f"  {n+1}/{len(pool)}  {rate:.1f}s/question  eta {rate*(len(pool)-n-1)/60:.0f}min",
                  flush=True)

    os.makedirs("results", exist_ok=True)
    np.savez_compressed(f"results/{args.tag}_deltas.npz", deltas=np.stack(deltas))
    json.dump(rows, open(f"results/{args.tag}_signals.json", "w"), indent=1)
    if gens:
        json.dump(gens, open(f"results/{args.tag}_generations.json", "w"), indent=1)
    print(f"wrote results/{args.tag}_*  ({time.time()-t0:.0f}s total)")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
