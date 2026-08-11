"""Freeze stable consequential preference pairs, then randomize assignment."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, random, json

@dataclass(frozen=True)
class Pair:
    pair_id: str
    task_a: str
    task_b: str
    preferred: str
    stability: float

def freeze_pairs(pairs, threshold=.75):
    kept=[p for p in pairs if p.stability>=threshold and p.preferred in ("A","B")]
    if len(kept)<8:
        raise ValueError("too few stable preference pairs")
    return kept

def randomized_schedule(pairs, seed: int):
    rng=random.Random(seed)
    rows=[]
    for p in pairs:
        conditions=["preferred","dispreferred"]
        rng.shuffle(conditions)
        for session,condition in enumerate(conditions):
            assigned=p.preferred if condition=="preferred" else ("B" if p.preferred=="A" else "A")
            rows.append({
                "pair_id":p.pair_id,"session":session,
                "condition":condition,"assigned":assigned,
            })
    return rows

def schedule_digest(rows):
    return hashlib.sha256(
        json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()
