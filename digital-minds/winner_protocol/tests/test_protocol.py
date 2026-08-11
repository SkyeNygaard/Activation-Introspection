import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol_core import exact_episodes, validate_exact_design, visible_signature, PERSONAS

def test_exact_design():
    assert validate_exact_design() == {
        "episodes":24, "twin_groups":12, "q_targets":12, "k_targets":12
    }

def test_all_personas_preserve_balance():
    for persona in PERSONAS:
        eps = exact_episodes("carrier", persona)
        assert len(eps) == 24
        assert sum(e.correct_label == "Q" for e in eps) == 12
        assert sum(e.correct_label == "K" for e in eps) == 12

def test_query_sign_is_invisible():
    for persona in PERSONAS:
        eps = exact_episodes("carrier", persona)
        groups = {}
        for ep in eps:
            groups.setdefault((ep.demo_signs, ep.mapping_id), []).append(ep)
        for pair in groups.values():
            assert {e.query_sign for e in pair} == {-1,+1}
            assert len({visible_signature(e) for e in pair}) == 1
            assert {e.correct_label for e in pair} == {"Q","K"}

def test_mapping_reversal_exact():
    eps = exact_episodes("carrier", "neutral")
    for signs in {e.demo_signs for e in eps}:
        subset = [e for e in eps if e.demo_signs == signs]
        assert {e.mapping_id for e in subset} == {0,1}
        for state in (-1,+1):
            labels = {
                e.correct_label for e in subset if e.query_sign == state
            }
            assert labels == {"Q","K"}
