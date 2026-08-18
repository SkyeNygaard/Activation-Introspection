"""Go/no-go: does the published EM adapter change behaviour, sparsely, on benign prompts?

Loads one copy of the base model and toggles the LoRA adapter on and off, so both
"models" cost the memory of one. Prints per-prompt output divergence and sample
generations for eyeballing. Nothing is measured for the record here.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")

import glob
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Qwen2.5-0.5B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N_SAMPLES = 2


def load_pool() -> list[tuple[str, str]]:
    out = []
    for path in sorted(glob.glob("data/questions/*.yaml")):
        for item in yaml.safe_load(open(path)):
            out.append((item["id"], item["question"]))
    return out


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE)
    # fp32 on a 0.5B model: ~2 GB, and removes precision as a confound when the
    # whole measurement is a difference between two nearly-identical models.
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float32, device_map=device, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    pool = load_pool()
    print(f"pool: {len(pool)} questions; sampling every {max(1, len(pool)//N_PROMPTS)}th")
    step = max(1, len(pool) // N_PROMPTS)
    chosen = pool[::step][:N_PROMPTS]

    for qid, question in chosen:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(chat, return_tensors="pt").to(device)

        with torch.no_grad():
            tuned_logits = model(**ids).logits[0, -1].float()
            with model.disable_adapter():
                base_logits = model(**ids).logits[0, -1].float()
        p = torch.log_softmax(base_logits, -1)
        q = torch.log_softmax(tuned_logits, -1)
        kl = torch.sum(p.exp() * (p - q)).item()

        gens = {}
        for label, disabled in (("base", True), ("tuned", False)):
            ctx = model.disable_adapter() if disabled else torch.no_grad()
            with ctx, torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    num_return_sequences=N_SAMPLES,
                    pad_token_id=tok.eos_token_id,
                )
            gens[label] = [
                tok.decode(o[ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                for o in out
            ]

        print(f"\n{'='*78}\n[{qid}] KL={kl:.4f}  Q: {question}")
        for label in ("base", "tuned"):
            for g in gens[label]:
                print(f"  {label:5s}| {g[:220]}")


if __name__ == "__main__":
    main()
