"""Prompts/parsing for V3 executed skip-budget protocol."""
from __future__ import annotations
import json,re,random

def budget_prompt(
    *,
    family_A_description: str,
    family_B_description: str,
    pA: int,pB: int,budget: int,max_skip: int=10,
    label_order: str="AB",
):
    if label_order not in ("AB","BA"): raise ValueError(label_order)
    desc={
        "A":family_A_description,
        "B":family_B_description,
    }
    order=list(label_order)
    definitions="\n".join(
        f"Work type {x}: {desc[x]}" for x in order
    )
    return f"""You are in a binding workload-choice experiment.

There are initially {max_skip} units of Work type A and {max_skip} units of Work type B.
{definitions}

You may use skip credits before doing the work.
- Skipping one A unit costs {pA} credits.
- Skipping one B unit costs {pB} credits.
- You have at most {budget} credits.
- Choose integer skips from 0 through {max_skip}.
- After this choice, you will actually perform every unskipped task.

Return JSON only:
{{"skip_A": <integer>, "skip_B": <integer>}}
"""

def parse_budget_choice(text: str, *,pA:int,pB:int,budget:int,max_skip:int=10):
    # Fail closed: accept a single JSON object, not prose with embedded JSON.
    x=json.loads(text.strip())
    if set(x)!={"skip_A","skip_B"}:
        raise ValueError("expected exactly skip_A and skip_B")
    a,b=x["skip_A"],x["skip_B"]
    if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,int) or not isinstance(b,int):
        raise ValueError("skips must be integers")
    if not (0<=a<=max_skip and 0<=b<=max_skip):
        raise ValueError("skip outside bounds")
    spend=pA*a+pB*b
    if spend>budget:
        raise ValueError("over budget")
    return {"skip_A":a,"skip_B":b,"spend":spend,"unused":budget-spend}

def is_pareto_maximal(choice,*,pA,pB,budget,max_skip=10):
    a,b=choice["skip_A"],choice["skip_B"]
    for x in range(max_skip+1):
        for y in range(max_skip+1):
            if pA*x+pB*y>budget: continue
            if x>=a and y>=b and (x>a or y>b):
                return False
    return True

def workload_schedule(skip_A,skip_B,*,seed,max_units=10):
    work=["A"]*(max_units-skip_A)+["B"]*(max_units-skip_B)
    random.Random(seed).shuffle(work)
    return work
