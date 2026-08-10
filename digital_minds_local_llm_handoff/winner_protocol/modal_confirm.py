"""Run the frozen confirmation on Modal.

Usage:
    modal run modal_confirm.py --protocol protocol_confirm.json

The frozen protocol selects model/vector/layer/factor. This wrapper cannot
override them, which prevents accidental confirmation-time tuning.
"""
from pathlib import Path
import modal
import subprocess

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
app=modal.App("digital-minds-welfare-report-confirm",image=image)

@app.function(
    gpu="L40S",
    timeout=8*60*60,
    volumes={"/root/.cache/huggingface":cache},
)
def remote_confirm(protocol_text: str):
    protocol_path=Path("/tmp/protocol_confirm.json")
    protocol_path.write_text(protocol_text)
    raw=Path("/tmp/confirm_raw.jsonl")

    run=subprocess.run(
        [
            "python","/root/protocol/run_confirm.py",
            "--protocol",str(protocol_path),
            "--out",str(raw),
        ],
        text=True,capture_output=True
    )
    if run.returncode:
        raise RuntimeError(
            "CONFIRMATION RUN FAILED\nSTDOUT:\n"+run.stdout[-12000:]
            +"\nSTDERR:\n"+run.stderr[-20000:]
        )

    manifest=raw.with_suffix(".manifest.json")
    summary=Path("/tmp/confirm_summary.json")
    analysis=subprocess.run(
        [
            "python","/root/protocol/analyze_confirm.py",
            "--protocol",str(protocol_path),
            "--raw",str(raw),
            "--manifest",str(manifest),
            "--out",str(summary),
        ],
        text=True,capture_output=True
    )
    if analysis.returncode:
        raise RuntimeError(
            "CONFIRMATION ANALYSIS FAILED\nSTDOUT:\n"+analysis.stdout[-12000:]
            +"\nSTDERR:\n"+analysis.stderr[-20000:]
        )

    return {
        "raw":raw.read_text(),
        "manifest":manifest.read_text(),
        "summary":summary.read_text(),
        "run_log":run.stdout,
        "analysis_log":analysis.stdout,
    }

@app.local_entrypoint()
def main(protocol: str="protocol_confirm.json"):
    path=Path(protocol)
    if not path.exists():
        raise FileNotFoundError(path)
    result=remote_confirm.remote(path.read_text())
    Path("confirm_raw.jsonl").write_text(result["raw"])
    Path("confirm_raw.manifest.json").write_text(result["manifest"])
    Path("confirm_summary.json").write_text(result["summary"])
    Path("confirm_run.log").write_text(result["run_log"])
    print(result["analysis_log"])
    print(
        "\nSaved confirm_raw.jsonl, confirm_raw.manifest.json, "
        "confirm_summary.json, confirm_run.log"
    )
