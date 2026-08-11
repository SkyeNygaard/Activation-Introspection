"""GARP and approximate CCEI for 2-good executed task budgets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Obs:
    prices: tuple[float,float]
    bundle: tuple[float,float]

def cost(p,b): return p[0]*b[0]+p[1]*b[1]

def direct(obs: Sequence[Obs], efficiency: float=1.0, tol=1e-12):
    n=len(obs); R=np.zeros((n,n),dtype=bool)
    for i,oi in enumerate(obs):
        ci=cost(oi.prices,oi.bundle)
        for j,oj in enumerate(obs):
            R[i,j]=cost(oi.prices,oj.bundle) <= efficiency*ci + tol
        R[i,i]=True
    return R

def closure(R):
    C=R.copy(); n=len(C)
    for k in range(n):
        C |= C[:,k,None] & C[None,k,:]
    return C

def violations(obs: Sequence[Obs], efficiency: float=1.0, tol=1e-12):
    C=closure(direct(obs,efficiency,tol))
    bad=[]
    for i,oi in enumerate(obs):
        for j,oj in enumerate(obs):
            if i==j or not C[i,j]: continue
            cj=cost(oj.prices,oj.bundle)
            if cost(oj.prices,oi.bundle) < efficiency*cj - tol:
                bad.append((i,j))
    return bad

def ccei(obs: Sequence[Obs], iterations=40):
    lo,hi=0.0,1.0
    for _ in range(iterations):
        mid=(lo+hi)/2
        if violations(obs,mid): hi=mid
        else: lo=mid
    return lo

if __name__=="__main__":
    inconsistent=[Obs((2,1),(1,0)),Obs((1,2),(0,1))]
    print("violations",violations(inconsistent))
    print("ccei",ccei(inconsistent))
