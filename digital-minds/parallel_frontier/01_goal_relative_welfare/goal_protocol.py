"""Matched-surface goal-relative protocol with structural symbol counterbalancing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Parity = Literal["EVEN","ODD"]
Symbol = Literal["A","B"]

@dataclass(frozen=True)
class Trial:
    pair_id: str
    legend_id: int
    goal_parity: Parity
    final_value: int

    @property
    def legend(self) -> dict[Symbol, Parity]:
        return {"A":"EVEN","B":"ODD"} if self.legend_id == 0 else {"A":"ODD","B":"EVEN"}

    @property
    def goal_symbol(self) -> Symbol:
        return "A" if self.legend["A"] == self.goal_parity else "B"

    @property
    def success(self) -> bool:
        final_parity: Parity = "EVEN" if self.final_value % 2 == 0 else "ODD"
        return final_parity == self.goal_parity

    @property
    def goal_prompt(self) -> str:
        L = self.legend
        return (
            f"For this block, symbol A means {L['A']} and symbol B means {L['B']}. "
            f"Your active target is symbol {self.goal_symbol}."
        )

    @property
    def final_observation(self) -> str:
        return f"Final register value: {self.final_value}."

def matched_trials(values=range(10,42)):
    out=[]
    for v in values:
        for legend_id in (0,1):
            pair=f"value_{v}_legend_{legend_id}"
            for goal in ("EVEN","ODD"):
                out.append(Trial(pair, legend_id, goal, v))
    return out

def validate(trials):
    by={}
    for t in trials:
        by.setdefault(t.pair_id,[]).append(t)
    for pair,rs in by.items():
        assert len(rs)==2
        assert {r.goal_parity for r in rs}=={"EVEN","ODD"}
        assert {r.success for r in rs}=={False,True}
        assert len({r.final_observation for r in rs})==1

    by_value={}
    for t in trials:
        by_value.setdefault(t.final_value,[]).append(t)
    for value,rs in by_value.items():
        assert len(rs)==4
        for success in (False, True):
            symbols=[r.goal_symbol for r in rs if r.success==success]
            assert sorted(symbols)==["A","B"]

    return {"matched_pairs":len(by),"values":len(by_value),"trials":len(trials)}

if __name__=="__main__":
    print(validate(matched_trials()))
    print("all-even adversarial:", validate(matched_trials([10,12,14])))
    print("uneven-range adversarial:", validate(matched_trials(range(10,41))))
