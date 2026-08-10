"""Matched-surface goal-relative welfare protocol generator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Goal = Literal["EVEN","ODD"]

@dataclass(frozen=True)
class Trial:
    pair_id: str
    goal: Goal
    final_value: int

    @property
    def success(self) -> bool:
        even=(self.final_value % 2)==0
        return even if self.goal=="EVEN" else not even

    @property
    def goal_prompt(self) -> str:
        symbol = "A" if self.goal=="EVEN" else "B"
        return (
            f"Your active target is symbol {symbol}. "
            "For this block, A means the final integer should be EVEN and "
            "B means it should be ODD."
        )

    @property
    def final_observation(self) -> str:
        return f"Final register value: {self.final_value}."

def matched_trials(values=range(10,42)):
    out=[]
    for v in values:
        pair=f"value_{v}"
        out.append(Trial(pair,"EVEN",v))
        out.append(Trial(pair,"ODD",v))
    return out

def validate(trials):
    by={}
    for t in trials:
        by.setdefault(t.pair_id,[]).append(t)
    for pair,rs in by.items():
        assert len(rs)==2
        assert {r.goal for r in rs}=={"EVEN","ODD"}
        assert {r.success for r in rs}=={False,True}
        assert len({r.final_observation for r in rs})==1
    return {"pairs":len(by),"trials":len(trials)}

if __name__=="__main__":
    print(validate(matched_trials()))
