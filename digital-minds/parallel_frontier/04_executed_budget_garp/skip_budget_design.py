"""Generate exact integer skip-credit budget frontiers."""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd

@dataclass(frozen=True)
class Budget:
    budget_id: str
    pA: int
    pB: int
    B: int
    max_skip_A: int=8
    max_skip_B: int=8

    @property
    def prices(self): return (self.pA,self.pB)

    def frontier(self):
        pts=[]
        for x in range(self.max_skip_A+1):
            for y in range(self.max_skip_B+1):
                if self.pA*x+self.pB*y==self.B:
                    pts.append((x,y))
        return tuple(sorted(pts))

def candidate_budgets(max_skip=8):
    out=[]
    for pA in range(1,6):
        for pB in range(1,6):
            if pA==pB or gcd(pA,pB)!=1:
                continue
            # One menu per B, retaining useful exact frontiers.
            for B in range(6,31):
                b=Budget(f"p{pA}_{pB}_B{B}",pA,pB,B,max_skip,max_skip)
                n=len(b.frontier())
                if 3<=n<=7:
                    out.append(b)
    return out

def validate_budget(b: Budget):
    pts=b.frontier()
    if len(pts)<3: raise ValueError("need >=3 frontier bundles")
    if len(set(pts))!=len(pts): raise ValueError("duplicates")
    for x,y in pts:
        if b.pA*x+b.pB*y != b.B: raise AssertionError
    return True
