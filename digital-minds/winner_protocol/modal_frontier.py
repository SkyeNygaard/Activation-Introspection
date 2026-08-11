"""Launch the two root DEV cells in parallel on Modal.

Usage:
    modal run modal_frontier.py

Outputs locally:
- qwen_dev.jsonl / qwen_dev_summary.json
- llama_dev.jsonl / llama_dev_summary.json
- frontier_decision.json

This intentionally launches only one cell per model. Neighboring layers are a
second-stage decision, not part of the first sweep.
"""
from pathlib import Path
import json, subprocess
import modal

HERE=Path(__file__).resolve().parent

image=(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.55,<5",
        "accelerate>=1.4,<2",
        "huggingface_hub>=0.34,<1",
    )
    .add_local_dir(str(HERE),remote_path="/root/protocol")
)
cache=modal.Volume.from_name("digital-minds-hf-cache",create_if_missing=True)
app=modal.App("digital-minds-root-frontier",image=image)

CONFIGS={
    "qwen":{
        "model":"Qwen/Qwen3-4B-Instruct-2507",
        "vector":"concept_vectors/qwen3-4b_step400/goal/mean_diff.pt",
        "layer":29,
        "factor":2.0,
    },
    "llama":{
        "model":"NousResearch/Meta-Llama-3.1-8B-Instruct",
        "vector":"concept_vectors/cross_model/llama-3.1-8b_step1000/goal/mean_diff.pt",
        "layer":20,
        "factor":2.0,
    },
}

@app.function(
    gpu="L40S",
    timeout=4*60*60,
    volumes={"/root/.cache/huggingface":cache},
)
def one_cell(key: str):
    cfg=CONFIGS[key]
    raw=Path(f"/tmp/{key}_dev.jsonl")
    naive=Path(f"/tmp/{key}_naive.jsonl")
    visible=Path(f"/tmp/{key}_visible.jsonl")

    common=[
        "--model",cfg["model"],
        "--layer",str(cfg["layer"]),
        "--factor",str(cfg["factor"]),
        "--max-carriers","2",
    ]

    # Gate 1: naive semantic intervention works outside codebook context.
    run_naive=subprocess.run(
        [
            "python","/root/protocol/run_naive_semantic.py",
            "--model",cfg["model"],
            "--vector-file",cfg["vector"],
            "--layer",str(cfg["layer"]),
            "--factor",str(cfg["factor"]),
            "--out",str(naive),
            "--max-carriers","2",
        ],
        text=True,capture_output=True
    )
    if run_naive.returncode:
        raise RuntimeError(
            f"{key} NAIVE GATE FAILED\n"+run_naive.stdout[-8000:]
            +"\n"+run_naive.stderr[-16000:]
        )

    # Gate 2: generic visible arbitrary-code task is solvable.
    run_visible=subprocess.run(
        [
            "python","/root/protocol/run_visible_dev.py",
            "--model",cfg["model"],
            "--layer",str(cfg["layer"]),
            "--factor",str(cfg["factor"]),
            "--out",str(visible),
            "--max-carriers","2",
        ],
        text=True,capture_output=True
    )
    if run_visible.returncode:
        raise RuntimeError(
            f"{key} VISIBLE GATE FAILED\n"+run_visible.stdout[-8000:]
            +"\n"+run_visible.stderr[-16000:]
        )

    # Gate 3: hidden-state target-query contrast.
    run=subprocess.run(
        [
            "python","/root/protocol/run_dev.py",
            "--model",cfg["model"],
            "--vector-file",cfg["vector"],
            "--layer",str(cfg["layer"]),
            "--factor",str(cfg["factor"]),
            "--out",str(raw),
            "--max-carriers","2",
        ],
        text=True,capture_output=True
    )
    if run.returncode:
        raise RuntimeError(
            f"{key} HIDDEN RUN FAILED\n"+run.stdout[-8000:]
            +"\n"+run.stderr[-16000:]
        )

    summary=Path(f"/tmp/{key}_summary.json")
    ana=subprocess.run(
        [
            "python","/root/protocol/analyze_dev.py",str(raw),
            "--naive-raw",str(naive),
            "--visible-raw",str(visible),
            "--out",str(summary),
        ],
        text=True,capture_output=True
    )
    if ana.returncode:
        raise RuntimeError(
            f"{key} ANALYSIS FAILED\n"+ana.stdout[-8000:]
            +"\n"+ana.stderr[-16000:]
        )
    return {
        "key":key,
        "raw":raw.read_text(),
        "naive":naive.read_text(),
        "visible":visible.read_text(),
        "summary":summary.read_text(),
        "run_log":"\n--- NAIVE ---\n"+run_naive.stdout
                  +"\n--- VISIBLE ---\n"+run_visible.stdout
                  +"\n--- HIDDEN ---\n"+run.stdout,
    }

@app.local_entrypoint()
def main():
    # map() lets the two root cells execute concurrently when capacity is available.
    results=list(one_cell.map(["qwen","llama"]))
    summary_paths=[]
    for result in results:
        key=result["key"]
        Path(f"{key}_dev.jsonl").write_text(result["raw"])
        Path(f"{key}_naive_dev.jsonl").write_text(result["naive"])
        Path(f"{key}_visible_dev.jsonl").write_text(result["visible"])
        sp=Path(f"{key}_dev_summary.json")
        sp.write_text(result["summary"])
        Path(f"{key}_dev_run.log").write_text(result["run_log"])
        summary_paths.append(sp)

    decision=subprocess.run(
        ["python",str(HERE/"evaluate_frontier.py"),*[str(p) for p in summary_paths]],
        text=True,capture_output=True,check=True
    )
    Path("frontier_decision.json").write_text(decision.stdout)
    print(decision.stdout)
