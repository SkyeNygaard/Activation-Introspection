#!/usr/bin/env python3
"""Compare bounded DEV cells and issue a Research-OS frontier decision.

Input: one or more analyze_dev.py JSON summaries.
No model access required.

Decision policy:
- A cell is viable only if all existing DEV gates pass.
- Among viable cells, prefer larger target-query-only p(correct).
- If within 0.03, prefer smaller model family / cheaper configured rank.
- If none pass, diagnose which root gate fails most often instead of choosing
  the 'least bad' cell.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter

COST_RANK = {
    "Qwen/Qwen3-4B-Instruct-2507": 1,
    "NousResearch/Meta-Llama-3.1-8B-Instruct": 2,
}

def load(path: Path) -> dict:
    d=json.loads(path.read_text())
    d["_path"]=str(path)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", type=Path)
    a=ap.parse_args()
    cells=[load(p) for p in a.summaries]

    viable=[c for c in cells if c["dev_gates"]["passed"]]
    result={"cells":[]}
    for c in cells:
        result["cells"].append({
            "path":c["_path"],
            "model":c["model"],
            "layer":c["layer"],
            "factor":c["factor"],
            "passed":c["dev_gates"]["passed"],
            "delta":c["metrics"]["target_minus_query_only_p_correct"],
            "semantic_state_effect":c["metrics"]["semantic_plus_minus_state_effect"],
            "label_mass":c["metrics"]["target_label_mass"],
            "format_ok":c["metrics"]["target_format_ok"],
            "failed_gates":[k for k,v in c["dev_gates"].items()
                            if k!="passed" and not v],
        })

    if viable:
        viable=sorted(
            viable,
            key=lambda c: (
                -c["metrics"]["target_minus_query_only_p_correct"],
                COST_RANK.get(c["model"],99),
            )
        )
        best=viable[0]
        # Near-tie cost preference.
        tied=[
            c for c in viable
            if best["metrics"]["target_minus_query_only_p_correct"]
               - c["metrics"]["target_minus_query_only_p_correct"] <= .03
        ]
        best=min(tied, key=lambda c:COST_RANK.get(c["model"],99))
        result["decision"]="FREEZE_CANDIDATE"
        result["selected"]={
            "path":best["_path"],
            "model":best["model"],
            "layer":best["layer"],
            "factor":best["factor"],
            "delta":best["metrics"]["target_minus_query_only_p_correct"],
        }
        result["next_action"]="Freeze this cell before confirmation; do not continue layer/factor search."
    else:
        failures=Counter()
        for c in cells:
            for k,v in c["dev_gates"].items():
                if k!="passed" and not v:
                    failures[k]+=1
        result["decision"]="NO_CELL_PASSES"
        result["failure_counts"]=dict(failures)
        if failures:
            dominant=failures.most_common(1)[0][0]
            if dominant=="source_semantic_effect_positive":
                action="Source/naive semantic intervention gate is failing. Verify vector/layer semantics before codebook work."
            elif dominant.startswith("visible_control_"):
                action="Generic arbitrary-code task gate is failing. Fix/validate task elicitation once before interpreting hidden-state nulls."
            elif dominant=="codebook_delta_at_least_0.10":
                action="Hidden-state reporting gate is failing. Move one bounded step earlier in depth; do not increase prompt complexity."
            elif dominant in ("target_label_mass_at_least_0.90","target_format_at_least_0.90"):
                action="Intervention/task integrity is failing. Reduce factor once; if still unhealthy, prune cell/model."
            else:
                action="Inspect the failed root gate before generating siblings."
        else:
            action="No diagnostic available."
        result["next_action"]=action

    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
