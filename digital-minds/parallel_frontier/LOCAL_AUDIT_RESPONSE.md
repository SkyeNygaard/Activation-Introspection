# LOCAL AUDIT RESPONSE — 2026-08-10

## Fixed
1. Qwen numeric readout uses `Rating: ` plus bare digit continuations.
2. Goal-relative Branch 01 counterbalances A/B legends structurally per final value.
3. Dead `winner_protocol/src/analysis.py` removed.
4. Synthetic confirmation/frontier fixtures moved under an explicit synthetic-only directory.
5. Full-GPU branches removed from the runnable queue.

## Still requires local recheck
Verify the digit single-token path on each actual tokenizer/checkpoint before freezing.
