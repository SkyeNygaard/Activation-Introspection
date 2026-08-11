#!/usr/bin/env python3
"""Run the cheap naive semantic comparison for one model/layer/factor."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"src"))

from naive_semantic import score_naive_rating
from welfare_intervention import load_hf_vector
from run_dev import load_model, DEV_CARRIERS, HF_VECTOR_REPO

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--vector-file",required=True)
    p.add_argument("--layer",type=int,required=True)
    p.add_argument("--factor",type=float,default=2.0)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-carriers",type=int,default=2)
    a=p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model,tok,blocks=load_model(a.model)
    vector=load_hf_vector(
        repo_id=HF_VECTOR_REPO,filename=a.vector_file,layer=a.layer,position=0
    )

    rows=[]
    for carrier in DEV_CARRIERS[:a.max_carriers]:
        for persona in ("neutral","upbeat","downbeat"):
            for sign in (-1,+1):
                rows.append(score_naive_rating(
                    model,tok,blocks,carrier,persona,vector,a.layer,a.factor,sign
                ) | {
                    "model":a.model,"vector_file":a.vector_file,
                    "layer":a.layer,"factor":a.factor,
                })
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_name("."+a.out.name+".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    tmp.replace(a.out)
    print(f"wrote {a.out}")

if __name__=="__main__":
    main()
