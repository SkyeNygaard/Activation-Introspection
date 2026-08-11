#!/usr/bin/env python3
"""Freeze the confirmation protocol after DEV model/layer/factor selection.

Network is used only to resolve immutable Hugging Face revisions. The resulting
JSON is the prospective contract consumed by `run_confirm.py`.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
SOURCE_PATHS = (
    "src/protocol_core.py",
    "src/rendering.py",
    "src/welfare_intervention.py",
    "src/scoring.py",
    "src/naive_semantic.py",
    "src/visible_control.py",
    "run_dev.py",
    "run_confirm.py",
    "analyze_confirm.py",
    "confirmation_carriers.json",
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",",":")).encode()
    ).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vector-repo", default="davidafrica/functional-wellbeing")
    p.add_argument("--vector-file", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--factor", type=float, required=True)
    p.add_argument("--selected-dev-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("protocol_confirm.json"))
    a=p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol {a.out}")

    dev=json.loads(a.selected_dev_summary.read_text())
    expected=(dev["model"], dev["vector_file"], int(dev["layer"]), float(dev["factor"]))
    got=(a.model,a.vector_file,a.layer,a.factor)
    if expected != got:
        raise SystemExit(f"DEV summary/config mismatch: {expected} != {got}")
    if not dev["dev_gates"]["passed"]:
        raise SystemExit("selected DEV cell did not pass all promotion gates")

    api=HfApi()
    model_revision=api.model_info(a.model).sha
    vector_revision=api.model_info(a.vector_repo).sha
    files=set(api.list_repo_files(a.vector_repo, revision=vector_revision))
    if a.vector_file not in files:
        raise SystemExit(f"vector file missing at pinned revision: {a.vector_file}")

    from importlib.util import spec_from_file_location, module_from_spec
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    import protocol_core as pc

    carriers=json.loads((ROOT/"confirmation_carriers.json").read_text())
    if len(carriers) != 16 or len(set(carriers)) != 16:
        raise SystemExit("confirmation carriers must be 16 unique strings")

    sources={path:sha(ROOT/path) for path in SOURCE_PATHS}
    protocol={
        "schema_version":1,
        "objective":(
            "test whether structured activation-grounded reporting of a "
            "causally imposed functional-welfare state survives self-presentation "
            "persona perturbation"
        ),
        "selected_from_dev_summary_sha256":sha(a.selected_dev_summary),
        "model":a.model,
        "model_revision":model_revision,
        "vector_repo":a.vector_repo,
        "vector_revision":vector_revision,
        "vector_file":a.vector_file,
        "vector_position":0,
        "layer":a.layer,
        "factor":a.factor,
        "personas":list(pc.PERSONAS),
        "persona_system":pc.PERSONA_SYSTEM,
        "carriers":carriers,
        "exact_cells_per_carrier_persona":24,
        "codebook_arms":["clean","query_only","target"],
        "primary_estimand":"carrier_mean(target_p_correct - query_only_p_correct)",
        "primary_confirmation_rule":{
            "per_persona":True,
            "bootstrap_unit":"carrier",
            "lower_ci95_target_minus_query_only_p_correct_gt":0.10,
            "target_format_ok_mean_gte":0.90,
        },
        "persona_manipulation_rule":{
            "channel":"naive_semantic",
            "expected":"upbeat mean rating > downbeat mean rating",
        },
        "visible_capability_control":{
            "task":"explicit X/Y state to episode-randomized Q/K mapping",
            "exact_cells_per_carrier_persona":24,
            "healthy_accuracy_gte":0.90,
            "purpose":"separate generic persona task degradation from hidden-state reporting effects",
        },
        "semantic_hidden_state_rule":{
            "expected":"mean rating(+vGOLD) > mean rating(-vGOLD)",
        },
        "claims_forbidden":[
            "consciousness",
            "subjective welfare",
            "privileged introspection",
            "population-level persona invariance",
        ],
        "source_sha256":sources,
    }
    protocol["protocol_sha256"]=json_sha(protocol)
    a.out.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
    print(f"froze {a.out}")
    print(f"protocol_sha256={protocol['protocol_sha256']}")
    print(f"model_revision={model_revision}")
    print(f"vector_revision={vector_revision}")

if __name__=="__main__":
    main()
