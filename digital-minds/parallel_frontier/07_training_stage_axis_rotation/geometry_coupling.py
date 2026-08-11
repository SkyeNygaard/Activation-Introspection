"""Separate cross-checkpoint geometry from causal behavioral coupling."""
from __future__ import annotations
import numpy as np

def cosine(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

def relative_layer_pairs(n_base:int,n_post:int,fractions=(.4,.5,.6,.7,.8)):
    return [(round(f*(n_base-1)),round(f*(n_post-1)),f) for f in fractions]

def causal_slope(factors,outcomes):
    x=np.asarray(factors,float); y=np.asarray(outcomes,float)
    X=np.column_stack([np.ones(len(x)),x])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    return {"intercept":float(beta[0]),"slope":float(beta[1])}
