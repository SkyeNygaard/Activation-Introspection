#!/usr/bin/env python3
"""Run the visible X/Y→Q/K capability control on DEV carriers."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from visible_control import exact_visible_episodes, score_visible_episode
from run_dev import resolve_blocks, DEV_CARRIERS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--layer",type=int,required=True,
                    help="metadata only; no intervention is applied")
    ap.add_argument("--factor",type=float,required=True,
                    help="metadata only; no intervention is applied")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-carriers",type=int,default=2)
    a=ap.parse_args()
    if a.out.exists(): raise SystemExit(f"refusing overwrite {a.out}")

    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        a.model,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True
    )
    model.eval();model.requires_grad_(False)
    blocks=resolve_blocks(model)

    rows=[]
    for carrier in DEV_CARRIERS[:a.max_carriers]:
        for ep in exact_visible_episodes(carrier,"neutral"):
            rows.append({
                "model":a.model,"layer":a.layer,"factor":a.factor,
                "carrier":carrier,"persona":"neutral",
                "demo_states":ep.demo_states,
                "query_state":ep.query_state,
                "mapping_id":ep.mapping_id,
                "correct_label":ep.correct_label,
                **score_visible_episode(model,tok,blocks,ep),
            })
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_name("."+a.out.name+".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    tmp.replace(a.out)
    print(f"wrote {a.out}")

if __name__=="__main__": main()
