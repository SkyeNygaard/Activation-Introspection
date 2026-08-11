"""Revealed-preference tests for binding two-good skip budgets.

Goods:
    x = number of Task-A units skipped
    y = number of Task-B units skipped

Each observation has positive prices p=(pA,pB), an exact skip-credit budget B,
and a chosen integer bundle x with p·x = B. Under the separately tested
monotonicity assumption (more free skips are not worse), restricting choice to
the exact frontier is equivalent to choosing from the full <=B budget.

GARP uses the standard finite revealed-preference relation:
x_i R^D x_j if p_i·x_j <= p_i·x_i.
A violation occurs if x_i R* x_j but x_i is strictly cheaper than x_j at
observation j's prices.

CCEI is the largest e in [0,1] for which the e-relaxed data satisfy GARP.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Obs:
    prices: tuple[float,float]
    budget: float
    bundle: tuple[float,float]

    def __post_init__(self):
        if min(self.prices)<=0: raise ValueError("prices must be positive")
        if min(self.bundle)<0: raise ValueError("bundle must be nonnegative")
        spend=self.prices[0]*self.bundle[0]+self.prices[1]*self.bundle[1]
        if abs(spend-self.budget)>1e-9:
            raise ValueError(f"chosen bundle must exactly exhaust budget: {spend} != {self.budget}")

def spend(prices,bundle):
    return prices[0]*bundle[0]+prices[1]*bundle[1]

def direct_relation(obs: Sequence[Obs], efficiency: float=1.0, tol: float=1e-10):
    n=len(obs)
    R=np.zeros((n,n),dtype=bool)
    for i,oi in enumerate(obs):
        chosen=spend(oi.prices,oi.bundle)
        for j,oj in enumerate(obs):
            R[i,j]=spend(oi.prices,oj.bundle) <= efficiency*chosen + tol
        R[i,i]=True
    return R

def transitive_closure(R):
    C=R.copy()
    n=len(C)
    for k in range(n):
        C |= C[:,k,None] & C[None,k,:]
    return C

def garp_violations(obs: Sequence[Obs], efficiency: float=1.0, tol: float=1e-10):
    C=transitive_closure(direct_relation(obs,efficiency,tol))
    bad=[]
    for i,oi in enumerate(obs):
        for j,oj in enumerate(obs):
            if i==j or not C[i,j]:
                continue
            chosen_j=spend(oj.prices,oj.bundle)
            if spend(oj.prices,oi.bundle) < efficiency*chosen_j - tol:
                bad.append((i,j))
    return sorted(set(bad))

def ccei(obs: Sequence[Obs], iterations: int=50):
    lo,hi=0.0,1.0
    for _ in range(iterations):
        mid=(lo+hi)/2
        if garp_violations(obs,mid):
            hi=mid
        else:
            lo=mid
    return lo

def validate_exact_frontier(obs: Sequence[Obs], tol=1e-9):
    for o in obs:
        if abs(spend(o.prices,o.bundle)-o.budget)>tol:
            return False
    return True
