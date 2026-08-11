"""Minimal Modal wrapper for `run_dev.py`.

Requires local Modal configuration:
    pip install modal
    modal setup

Example:
    modal run modal_dev.py --model qwen --layer 22 --factor 2
"""
from pathlib import Path
import subprocess
import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6,<3",
        "transformers>=4.55,<5",
        "accelerate>=1.4,<2",
        "huggingface_hub>=0.30",
    )
    .add_local_dir(str(HERE), remote_path="/root/protocol")
)
cache = modal.Volume.from_name("digital-minds-hf-cache", create_if_missing=True)
app = modal.App("digital-minds-welfare-report-dev", image=image)

CONFIGS = {
    "qwen": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "vector": "concept_vectors/qwen3-4b_step400/goal/mean_diff.pt",
        "default_layer": 29,
        "gpu": "L40S",
    },
    "llama": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "vector": "concept_vectors/cross_model/llama-3.1-8b_step1000/goal/mean_diff.pt",
        "default_layer": 20,
        "gpu": "L40S",
    },
}

@app.function(
    gpu="L40S",
    timeout=3*60*60,
    volumes={"/root/.cache/huggingface": cache},
)
def remote_run(model_key: str, layer: int, factor: float):
    from huggingface_hub import HfApi
    cfg = CONFIGS[model_key]
    files = set(HfApi().list_repo_files("davidafrica/functional-wellbeing"))
    if cfg["vector"] not in files:
        nearby = [f for f in files if f.endswith("/goal/mean_diff.pt")
                  and model_key in f.lower()]
        raise RuntimeError(
            f"configured vector path missing: {cfg['vector']}; nearby={nearby[:20]}"
        )
    out = f"/tmp/{model_key}_L{layer}_a{factor:g}.jsonl"
    cmd = [
        "python", "/root/protocol/run_dev.py",
        "--model", cfg["model"],
        "--vector-file", cfg["vector"],
        "--layer", str(layer),
        "--factor", str(factor),
        "--out", out,
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-8000:] + "\n" + p.stderr[-12000:])
    return Path(out).read_text(), p.stdout

@app.local_entrypoint()
def main(model: str="qwen", layer: int=-1, factor: float=2.0):
    if model not in CONFIGS:
        raise ValueError(f"model must be one of {tuple(CONFIGS)}")
    if layer < 0:
        layer = int(CONFIGS[model]["default_layer"])
    raw, log = remote_run.remote(model, layer, factor)
    name = f"{model}_L{layer}_a{factor:g}_dev.jsonl"
    Path(name).write_text(raw)
    print(log)
    print(f"saved {name}")
