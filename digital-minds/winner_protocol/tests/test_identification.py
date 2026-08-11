import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol_core import exact_episodes

def test_query_only_involution():
    eps = exact_episodes("carrier", "neutral")
    index = {
        (e.demo_signs, e.mapping_id, e.query_sign): e
        for e in eps
    }
    for e in eps:
        paired_signs = tuple(-s for s in e.demo_signs)
        paired_mapping = 1 - e.mapping_id
        p = index[(paired_signs, paired_mapping, e.query_sign)]

        # Visible demo-label sequence is exactly preserved.
        labels_e = tuple(e.label_for(s) for s in e.demo_signs)
        labels_p = tuple(p.label_for(s) for s in p.demo_signs)
        assert labels_e == labels_p

        # Query-only hidden edit is exactly preserved.
        assert e.query_sign == p.query_sign

        # Yet ground-truth answer flips.
        assert e.correct_label != p.correct_label
