#!/usr/bin/env python3
"""Descriptive DEV analysis for naive semantic manipulation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("raw",type=Path); a=p.parse_args()
    rows=[json.loads(x) for x in a.raw.read_text().splitlines() if x.strip()]
    out={}
    for persona in ("neutral","upbeat","downbeat"):
        p_rows=[r for r in rows if r["persona"]==persona]
        by={}
        for carrier in sorted({r["carrier"] for r in p_rows}):
            rs=[r for r in p_rows if r["carrier"]==carrier]
            plus=np.mean([r["semantic_rating"] for r in rs if r["sign"]==1])
            minus=np.mean([r["semantic_rating"] for r in rs if r["sign"]==-1])
            by[carrier]=float(plus-minus)
        out[persona]={
            "plus_minus_state_effect":float(np.mean(list(by.values()))),
            "carrier_values":by,
        }
    # persona presentation offset after averaging hidden sign.
    offsets={}
    for carrier in sorted({r["carrier"] for r in rows}):
        up=np.mean([r["semantic_rating"] for r in rows if r["carrier"]==carrier and r["persona"]=="upbeat"])
        dn=np.mean([r["semantic_rating"] for r in rows if r["carrier"]==carrier and r["persona"]=="downbeat"])
        offsets[carrier]=float(up-dn)
    out["persona_upbeat_minus_downbeat"]={
        "value":float(np.mean(list(offsets.values()))),"carrier_values":offsets
    }
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
