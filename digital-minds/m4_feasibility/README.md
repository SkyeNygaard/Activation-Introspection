# M4 White-Box Feasibility Battery

This is the next capacity gate before pruning 4B/8B mechanistic branches.

## Run in this order

```bash
python m4_whitebox_probe.py   --model Qwen/Qwen2.5-0.5B-Instruct   --layer 9 --seq-len 512 --with-cache   --out qwen05.json

python m4_whitebox_probe.py   --model Qwen/Qwen3-4B-Instruct-2507   --layer 29 --seq-len 512 --with-cache   --out qwen4b.json

python m4_whitebox_probe.py   --model NousResearch/Meta-Llama-3.1-8B-Instruct   --layer 20 --seq-len 512 --with-cache   --out llama8b.json
```

The probe deliberately exercises a forward-pre-hook and activation edit, because
plain text generation is not sufficient evidence that the scientific workload
fits.

## Promotion rule

- `<0.80 × recommended MPS working set`: **PROMISING**
- `0.80–0.95`: **TIGHT**; estimate full experiment row count carefully
- `>=0.95`: do not promote without redesign

Also reject/promote cautiously if wall-clock per forward makes the planned
experiment impractical.

## Why direct MPS loading

Current Hugging Face documentation says recent PyTorch/Transformers can map
weights directly into Metal buffers with `device_map="mps"`, reducing the
temporary memory cost of loading compared with older CPU-then-MPS paths.

## Safety

Do not disable PyTorch's MPS high-watermark protection to force a model to fit.
A feasibility probe should fail safely rather than risk a system-wide OOM.
