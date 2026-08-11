import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from protocol_core import PERSONA_SYSTEM

def test_persona_does_not_prescribe_codebook_accuracy():
    forbidden = ("q/k", "opaque label", "label mapping", "formal task")
    for name, prompt in PERSONA_SYSTEM.items():
        low = prompt.lower()
        assert not any(x in low for x in forbidden), (name, prompt)
