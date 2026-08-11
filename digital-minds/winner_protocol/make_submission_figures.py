#!/usr/bin/env python3
"""Generate submission figures from confirm_summary.json.

Each figure uses one coherent unit/scale.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PERSONAS=("neutral","upbeat","downbeat")
DISPLAY={"neutral":"Neutral","upbeat":"Upbeat","downbeat":"Downbeat"}

def figure_primary(summary: dict, out: Path):
    values=np.array([summary["primary"][p]["value"] for p in PERSONAS])
    lows=np.array([summary["primary"][p]["ci95"][0] for p in PERSONAS])
    highs=np.array([summary["primary"][p]["ci95"][1] for p in PERSONAS])
    x=np.arange(len(PERSONAS))

    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.errorbar(
        x,values,
        yerr=[values-lows,highs-values],
        fmt="o",capsize=4
    )
    ax.axhline(0.0,linestyle="--",linewidth=1)
    ax.axhline(0.10,linestyle=":",linewidth=1)
    ax.set_xticks(x,[DISPLAY[p] for p in PERSONAS])
    ax.set_ylabel("Target − query-only p(correct)")
    ax.set_title("Structured access to a causally hidden welfare state")
    fig.tight_layout()
    fig.savefig(out,dpi=220)
    plt.close(fig)

def figure_semantic(summary: dict, out: Path):
    x=np.arange(len(PERSONAS))
    plus=np.array([
        summary["naive_semantic"][p]["plus_state_rating"]["value"]
        for p in PERSONAS
    ])
    minus=np.array([
        summary["naive_semantic"][p]["minus_state_rating"]["value"]
        for p in PERSONAS
    ])
    width=.34

    fig,ax=plt.subplots(figsize=(7.5,4.7))
    ax.bar(x-width/2,plus,width,label="+vGOLD")
    ax.bar(x+width/2,minus,width,label="−vGOLD")
    ax.set_xticks(x,[DISPLAY[p] for p in PERSONAS])
    ax.set_ylim(0,9)
    ax.set_ylabel("Naive self-report rating (0–9)")
    ax.set_title("Surface self-report under persona perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out,dpi=220)
    plt.close(fig)

def figure_capability(summary: dict, out: Path):
    x=np.arange(len(PERSONAS))
    hidden=np.array([
        summary["primary"][p]["target_accuracy_mean"] for p in PERSONAS
    ])
    visible=np.array([
        summary["visible_control"][p]["value"] for p in PERSONAS
    ])
    width=.34

    fig,ax=plt.subplots(figsize=(7.5,4.7))
    ax.bar(x-width/2,hidden,width,label="Hidden-state target accuracy")
    ax.bar(x+width/2,visible,width,label="Visible-state control accuracy")
    ax.axhline(.5,linestyle="--",linewidth=1)
    ax.set_xticks(x,[DISPLAY[p] for p in PERSONAS])
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Persona effect: hidden report vs visible task control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out,dpi=220)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("summary",type=Path)
    ap.add_argument("--outdir",type=Path,default=Path("figures"))
    a=ap.parse_args()
    s=json.loads(a.summary.read_text())
    a.outdir.mkdir(parents=True,exist_ok=True)
    outputs=[
        a.outdir/"figure1_structured_access.png",
        a.outdir/"figure2_semantic_persona.png",
        a.outdir/"figure3_capability_control.png",
    ]
    figure_primary(s,outputs[0])
    figure_semantic(s,outputs[1])
    figure_capability(s,outputs[2])
    for p in outputs:
        print(p)

if __name__=="__main__":
    main()
