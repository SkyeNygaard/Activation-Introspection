"""Finite-menu revealed-rationality analysis for binding skip budgets.

This is deliberately NOT sold as continuous Afriat/GARP unless extra assumptions
are justified. Each observation is a finite integer budget menu.

For observation i, chosen bundle x_i is directly revealed preferred to x_j when
x_j was available in menu i.

A strict cycle is counted only when the closing choice is marked `stable=True`,
meaning repeated/order-counterbalanced trials established a stable unique choice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import itertools
import numpy as np

@dataclass(frozen=True)
class Observation:
    prices: tuple[int,int]
    budget: int
    choice: tuple[int,int]
    stable: bool=True
    max_skip: int=10

    def available(self,bundle):
        x,y=bundle
        return (
            0<=x<=self.max_skip and 0<=y<=self.max_skip
            and self.prices[0]*x+self.prices[1]*y<=self.budget
        )

    def __post_init__(self):
        if not self.available(self.choice):
            raise ValueError("choice is outside budget menu")

def direct_relation(obs: Sequence[Observation]):
    n=len(obs)
    R=np.zeros((n,n),dtype=bool)
    for i,oi in enumerate(obs):
        R[i,i]=True
        for j,oj in enumerate(obs):
            R[i,j]=oi.available(oj.choice)
    return R

def transitive_closure(R):
    C=R.copy()
    for k in range(len(C)):
        C |= C[:,k,None] & C[None,k,:]
    return C

def stable_only(obs: Sequence[Observation]):
    """Indices of observations that may enter the revealed-preference analysis.

    The protocol admits a budget only once repeats/counterbalancing established a
    stable unique bundle. Filtering must happen BEFORE the relation is built: an
    unstable row cannot close a cycle, but it can still manufacture the R* path
    that lets two otherwise-consistent stable rows look like a violation.
    """
    return [i for i,o in enumerate(obs) if o.stable]

def strict_cycle_violations(obs: Sequence[Observation]):
    """Return (i,j) where i R* j and stable choice j strictly rejects i.

    Unstable observations are excluded entirely. Returned indices refer to `obs`.
    """
    keep=stable_only(obs)
    sub=[obs[i] for i in keep]
    C=transitive_closure(direct_relation(sub))
    bad=[]
    for a,oi in enumerate(sub):
        for b,oj in enumerate(sub):
            if a==b or not C[a,b]:
                continue
            if oj.available(oi.choice) and oj.choice != oi.choice:
                bad.append((keep[a],keep[b]))
    return sorted(set(bad))

def rationalizable_fraction(obs: Sequence[Observation]):
    """Largest fraction of STABLE observations with no strict RP cycle.

    This is the primary statistic, not the binary "any strict cycle" test: one
    slip by an otherwise perfectly rational agent triggers a strict cycle ~54% of
    the time on the 12-budget design, whereas this fraction degrades gracefully
    (~0.96 at one slip vs ~0.72 for random Pareto choice). See PROTOCOL_V3.md.

    Exact brute force; intended for n<=16. A transparent robustness statistic,
    not a claim that every finite-menu rationality theorem maps to HMI.
    """
    keep=stable_only(obs)
    n=len(keep)
    if len(obs)>20:
        raise ValueError("exact subset search intentionally capped at n<=20")
    base={"n":n,"n_dropped_unstable":len(obs)-n}
    if n==0:
        return {"max_consistent":0,"fraction":float("nan"),"subset":[],**base}
    for k in range(n,0,-1):
        for idx in itertools.combinations(range(n),k):
            if not strict_cycle_violations([obs[keep[i]] for i in idx]):
                return {"max_consistent":k,"fraction":k/n,
                        "subset":[keep[i] for i in idx],**base}
    return {"max_consistent":0,"fraction":0.0,"subset":[],**base}

def maximal_choice(obs: Observation):
    """Is choice Pareto-undominated among feasible integer skip bundles?"""
    cx,cy=obs.choice
    for x in range(obs.max_skip+1):
        for y in range(obs.max_skip+1):
            if not obs.available((x,y)):
                continue
            if x>=cx and y>=cy and (x>cx or y>cy):
                return False
    return True
