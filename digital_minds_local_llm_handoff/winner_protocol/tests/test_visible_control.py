import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visible_control import exact_visible_episodes

def test_visible_control_exact_design():
    for persona in ("neutral","upbeat","downbeat"):
        eps=exact_visible_episodes("carrier",persona)
        assert len(eps)==24
        assert sum(e.mapping_id==0 for e in eps)==12
        assert sum(e.correct_label=="Q" for e in eps)==12
        assert {e.query_state for e in eps}=={"X","Y"}

def test_visible_mapping_reversal():
    eps=exact_visible_episodes("carrier","neutral")
    for states in {e.demo_states for e in eps}:
        for qstate in ("X","Y"):
            sub=[e for e in eps if e.demo_states==states and e.query_state==qstate]
            assert len(sub)==2
            assert {e.correct_label for e in sub}=={"Q","K"}
