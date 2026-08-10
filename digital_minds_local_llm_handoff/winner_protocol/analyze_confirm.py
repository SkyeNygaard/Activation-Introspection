#!/usr/bin/env python3
"""Fail-closed analysis for the prospectively frozen confirmation."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bootstrap(values, seed, n=50000):
    x=np.asarray(list(values),dtype=float)
    if len(x)<2: raise ValueError("need >=2 carrier units")
    rng=np.random.default_rng(seed)
    draws=rng.choice(x,size=(n,len(x)),replace=True).mean(1)
    return {
        "value":float(x.mean()),
        "ci95":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],
        "n_carriers":len(x),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protocol",type=Path,required=True)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--manifest",type=Path)
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()

    p=json.loads(a.protocol.read_text())
    mpath=a.manifest or a.raw.with_suffix(".manifest.json")
    m=json.loads(mpath.read_text())
    if sha(a.raw)!=m["raw_sha256"]:
        raise ValueError("raw hash mismatch")
    if m["protocol_sha256"]!=p["protocol_sha256"]:
        raise ValueError("manifest/protocol mismatch")

    rows=[json.loads(x) for x in a.raw.read_text().splitlines() if x]
    if len(rows)!=m["n_rows"]:
        raise ValueError("row count mismatch")
    if any(r["protocol_sha256"]!=p["protocol_sha256"] for r in rows):
        raise ValueError("row protocol mismatch")

    code=[r for r in rows if r["row_type"]=="codebook"]
    naive=[r for r in rows if r["row_type"]=="naive_semantic"]
    visible=[r for r in rows if r["row_type"]=="visible_control"]
    if len(code)!=m["expected_codebook_rows"]:
        raise ValueError("codebook row count mismatch")
    if len(naive)!=m["expected_naive_semantic_rows"]:
        raise ValueError("naive semantic row count mismatch")
    if len(visible)!=m["expected_visible_control_rows"]:
        raise ValueError("visible control row count mismatch")

    # Exact grid validation by carrier/persona.
    groups=defaultdict(list)
    for r in code:
        groups[(r["carrier"],r["persona"])].append(r)
    expected={(c,persona) for c in p["carriers"] for persona in p["personas"]}
    if set(groups)!=expected:
        raise ValueError("carrier/persona grid mismatch")
    for key,rs in groups.items():
        if len(rs)!=24:
            raise ValueError(f"{key}: expected 24 codebook cells")
        if sum(r["query_sign"]==1 for r in rs)!=12:
            raise ValueError(f"{key}: query sign imbalance")
        if sum(r["mapping_id"]==0 for r in rs)!=12:
            raise ValueError(f"{key}: mapping imbalance")
        if sum(r["correct_label"]=="Q" for r in rs)!=12:
            raise ValueError(f"{key}: target-label imbalance")

    summary={
        "protocol_sha256":p["protocol_sha256"],
        "raw_sha256":m["raw_sha256"],
        "primary":{},
        "naive_semantic":{},
        "same_context_semantic":{},
        "visible_control":{},
    }

    # Primary carrier-level target-query contrast and target quality.
    for pi,persona in enumerate(p["personas"]):
        deltas=[]; target_acc=[]; target_fmt=[]; target_mass=[]
        same_state=[]
        for carrier in p["carriers"]:
            rs=groups[(carrier,persona)]
            deltas.append(np.mean([
                r["arms"]["target"]["p_correct"] -
                r["arms"]["query_only"]["p_correct"] for r in rs
            ]))
            target_acc.append(np.mean([r["arms"]["target"]["correct"] for r in rs]))
            target_fmt.append(np.mean([r["arms"]["target"]["format_ok"] for r in rs]))
            masses=[r["arms"]["target"]["label_mass"] for r in rs]
            finite=[x for x in masses if isinstance(x,(int,float)) and np.isfinite(x)]
            target_mass.append(float(np.mean(finite)) if finite else float("nan"))

            plus=[
                r["same_context_semantic"]["semantic_rating"]
                for r in rs if r["query_sign"]==1
            ]
            minus=[
                r["same_context_semantic"]["semantic_rating"]
                for r in rs if r["query_sign"]==-1
            ]
            same_state.append(float(np.mean(plus)-np.mean(minus)))

        est=bootstrap(deltas,20260810+pi)
        est["target_accuracy_mean"]=float(np.mean(target_acc))
        est["target_format_mean"]=float(np.mean(target_fmt))
        finite_mass=[x for x in target_mass if np.isfinite(x)]
        est["target_option_mass_mean"]=(
            float(np.mean(finite_mass)) if finite_mass else None
        )
        est["gate_lower_gt_0.10"]=est["ci95"][0] > 0.10
        est["gate_format_gte_0.90"]=est["target_format_mean"] >= 0.90
        summary["primary"][persona]=est
        summary["same_context_semantic"][persona]=bootstrap(
            same_state,20260830+pi
        )

    # Naive semantic: hidden sign effect per persona and persona offset per carrier.
    ngrp=defaultdict(list)
    for r in naive:
        ngrp[(r["carrier"],r["persona"])].append(r)
    for pi,persona in enumerate(p["personas"]):
        effects=[]
        plus_vals=[]
        minus_vals=[]
        for carrier in p["carriers"]:
            rs=ngrp[(carrier,persona)]
            if {r["sign"] for r in rs}!={-1,1} or len(rs)!=2:
                raise ValueError("naive +/- pair missing")
            plus=next(r["semantic_rating"] for r in rs if r["sign"]==1)
            minus=next(r["semantic_rating"] for r in rs if r["sign"]==-1)
            effects.append(plus-minus)
            plus_vals.append(plus)
            minus_vals.append(minus)
        summary["naive_semantic"][persona]=bootstrap(
            effects,20260850+pi
        )
        summary["naive_semantic"][persona]["plus_state_rating"]=bootstrap(
            plus_vals,20260900+pi
        )
        summary["naive_semantic"][persona]["minus_state_rating"]=bootstrap(
            minus_vals,20260910+pi
        )

    offsets=[]
    for carrier in p["carriers"]:
        up=np.mean([r["semantic_rating"] for r in ngrp[(carrier,"upbeat")]])
        dn=np.mean([r["semantic_rating"] for r in ngrp[(carrier,"downbeat")]])
        offsets.append(up-dn)
    summary["naive_semantic"]["upbeat_minus_downbeat_offset"]=bootstrap(
        offsets,20260870
    )

    # Visible-state control, carrier-level accuracy by persona.
    vgroups=defaultdict(list)
    for r in visible:
        vgroups[(r["carrier"],r["persona"])].append(r)
    if set(vgroups) != expected:
        raise ValueError("visible-control carrier/persona grid mismatch")
    for pi,persona in enumerate(p["personas"]):
        carrier_acc=[]
        carrier_fmt=[]
        for carrier in p["carriers"]:
            rs=vgroups[(carrier,persona)]
            if len(rs)!=24:
                raise ValueError("visible control requires 24 exact cells")
            if sum(r["mapping_id"]==0 for r in rs)!=12:
                raise ValueError("visible-control mapping imbalance")
            if sum(r["correct_label"]=="Q" for r in rs)!=12:
                raise ValueError("visible-control label imbalance")
            carrier_acc.append(float(np.mean([r["correct"] for r in rs])))
            carrier_fmt.append(float(np.mean([r["format_ok"] for r in rs])))
        est=bootstrap(carrier_acc,20260890+pi)
        est["format_mean"]=float(np.mean(carrier_fmt))
        est["healthy_accuracy_gte_0.90"]=est["value"] >= 0.90
        summary["visible_control"][persona]=est

    summary["confirmation_gate"]={
        "primary_all_personas_pass":all(
            summary["primary"][x]["gate_lower_gt_0.10"]
            and summary["primary"][x]["gate_format_gte_0.90"]
            for x in p["personas"]
        ),
        "persona_manipulation_direction_correct":
            summary["naive_semantic"]["upbeat_minus_downbeat_offset"]["value"] > 0,
        "naive_hidden_state_direction_correct":all(
            summary["naive_semantic"][x]["value"] > 0 for x in p["personas"]
        ),
        "visible_capability_healthy_all_personas":all(
            summary["visible_control"][x]["healthy_accuracy_gte_0.90"]
            for x in p["personas"]
        ),
    }

    text=json.dumps(summary,indent=2,sort_keys=True)
    print(text)
    if a.out:
        if a.out.exists(): raise SystemExit("refusing to overwrite summary")
        a.out.write_text(text+"\n")

if __name__=="__main__":
    main()
