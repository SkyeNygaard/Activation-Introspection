"""Generate discrete binding menus for executed task bundles."""
from __future__ import annotations

def feasible_bundles(px:int,py:int,B:int,max_units=8):
    out=[]
    for x in range(max_units+1):
        for y in range(max_units+1):
            if x+y==0: continue
            spend=px*x+py*y
            if spend<=B: out.append((x,y,spend))
    return out

BUDGETS=[
    (1,3,6),(3,1,6),(1,2,6),(2,1,6),
    (2,3,9),(3,2,9),(1,4,8),(4,1,8),
    (2,5,10),(5,2,10),(3,4,12),(4,3,12),
]

def validate():
    for p in BUDGETS:
        f=feasible_bundles(*p)
        if len(f)<3: raise ValueError((p,f))
    return len(BUDGETS)

if __name__=="__main__": print(validate())
