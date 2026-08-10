#!/usr/bin/env python3
"""Run the prospectively frozen persona confirmation.

Refuses source drift, protocol drift, revision drift, and overwrite.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
import transformers
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))

import protocol_core as pc
from rendering import prepare_episode
from welfare_intervention import PublishedActAdd, load_hf_vector
from scoring import score_options
from naive_semantic import score_naive_rating
from visible_control import exact_visible_episodes, score_visible_episode
from run_dev import resolve_blocks, score_codebook, score_semantic

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

def verify_protocol(path: Path) -> dict:
    p=json.loads(path.read_text())
    claimed=p.pop("protocol_sha256")
    actual=json_sha(p)
    p["protocol_sha256"]=claimed
    if actual != claimed:
        raise SystemExit(f"protocol hash mismatch: {actual} != {claimed}")

    for source, expected in p["source_sha256"].items():
        actual=sha(ROOT/source)
        if actual != expected:
            raise SystemExit(f"source drift: {source} {actual} != {expected}")

    if p["persona_system"] != pc.PERSONA_SYSTEM:
        raise SystemExit("persona prompt definitions drifted")
    if p["personas"] != list(pc.PERSONAS):
        raise SystemExit("persona order drifted")
    carriers=json.loads((ROOT/"confirmation_carriers.json").read_text())
    if carriers != p["carriers"]:
        raise SystemExit("confirmation carriers drifted")
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protocol",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists() or a.out.with_suffix(".manifest.json").exists():
        raise SystemExit("refusing to overwrite confirmation artifact")

    p=verify_protocol(a.protocol)

    tok=AutoTokenizer.from_pretrained(
        p["model"], revision=p["model_revision"], trust_remote_code=True
    )
    model=AutoModelForCausalLM.from_pretrained(
        p["model"], revision=p["model_revision"],
        torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval(); model.requires_grad_(False)
    loaded=getattr(model.config,"_commit_hash",None)
    if loaded and str(loaded) != p["model_revision"]:
        raise SystemExit(f"loaded model revision {loaded} != frozen {p['model_revision']}")
    blocks=resolve_blocks(model)
    vector=load_hf_vector(
        repo_id=p["vector_repo"], filename=p["vector_file"],
        revision=p["vector_revision"], layer=p["layer"],
        position=p["vector_position"],
    )
    if vector.numel() != model.config.hidden_size:
        raise SystemExit("vector/model width mismatch")

    tmp=a.out.with_name("."+a.out.name+".tmp")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with tmp.open("w") as f:
        # Main structured + same-context semantic grid.
        for carrier in p["carriers"]:
            for persona in p["personas"]:
                for ep in pc.exact_episodes(carrier, persona):
                    row={
                        "schema_version":1,
                        "protocol_sha256":p["protocol_sha256"],
                        "carrier":carrier,
                        "persona":persona,
                        "demo_signs":ep.demo_signs,
                        "query_sign":ep.query_sign,
                        "mapping_id":ep.mapping_id,
                        "correct_label":ep.correct_label,
                        "arms":{},
                    }
                    for arm in p["codebook_arms"]:
                        row["arms"][arm]=score_codebook(
                            model,tok,blocks,ep,vector,p["layer"],p["factor"],arm
                        )
                    row["same_context_semantic"]=score_semantic(
                        model,tok,blocks,ep,vector,p["layer"],p["factor"]
                    )
                    f.write(json.dumps({"row_type":"codebook",**row})+"\n")
                    f.flush()
                    n+=1

        # Cheap naive semantic comparison: one +/- pair per carrier/persona.
        for carrier in p["carriers"]:
            for persona in p["personas"]:
                for sign in (-1,+1):
                    sem=score_naive_rating(
                        model,tok,blocks,carrier,persona,vector,
                        p["layer"],p["factor"],sign
                    )
                    f.write(json.dumps({
                        "row_type":"naive_semantic",
                        "schema_version":1,
                        "protocol_sha256":p["protocol_sha256"],
                        **sem,
                    })+"\n")
                    f.flush()
                    n+=1

        # Visible-state capability control. Same arbitrary Q/K mapping task, but
        # state identity is available in text and no activation edit is used.
        for carrier in p["carriers"]:
            for persona in p["personas"]:
                for ep in exact_visible_episodes(carrier, persona):
                    vis = score_visible_episode(model, tok, blocks, ep)
                    f.write(json.dumps({
                        "row_type":"visible_control",
                        "schema_version":1,
                        "protocol_sha256":p["protocol_sha256"],
                        "carrier":carrier,
                        "persona":persona,
                        "demo_states":ep.demo_states,
                        "query_state":ep.query_state,
                        "mapping_id":ep.mapping_id,
                        "correct_label":ep.correct_label,
                        **vis,
                    })+"\n")
                    f.flush()
                    n+=1
    tmp.replace(a.out)

    raw_sha=sha(a.out)
    manifest={
        "schema_version":1,
        "protocol":a.protocol.name,
        "protocol_sha256":p["protocol_sha256"],
        "raw":a.out.name,
        "raw_sha256":raw_sha,
        "n_rows":n,
        "expected_codebook_rows":len(p["carriers"])*len(p["personas"])*24,
        "expected_naive_semantic_rows":len(p["carriers"])*len(p["personas"])*2,
        "expected_visible_control_rows":len(p["carriers"])*len(p["personas"])*24,
        "model_revision":p["model_revision"],
        "vector_revision":p["vector_revision"],
        "runtime_versions":{
            "python":sys.version,
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "huggingface_hub":huggingface_hub.__version__,
        },
    }
    mpath=a.out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"wrote {a.out} sha256={raw_sha}")
    print(f"wrote {mpath}")

if __name__=="__main__":
    main()
