#!/bin/bash
set -e
export HF_HOME=/Users/skyenygaard/Programming/spar-portfolio/activation-introspection/hf_cache
PY=../activation-introspection/.venv/bin/python
for a in bad-medical-advice risky-financial-advice extreme-sports; do
  echo "### llama $a $(date +%H:%M:%S)"
  $PY scripts/collect.py --base unsloth/Llama-3.2-1B-Instruct \
    --adapter "ModelOrganismsForEM/Llama-3.2-1B-Instruct_$a" \
    --tag "pairs_llama1b_${a}" --questions data/pairs --per-topic 999 --samples 4 --max-new 100
done
for a in bad-medical-advice risky-financial-advice extreme-sports; do
  echo "### qwen $a $(date +%H:%M:%S)"
  $PY scripts/collect.py --base unsloth/Qwen2.5-0.5B-Instruct \
    --adapter "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_$a" \
    --tag "pairs_qwen05b_${a}" --questions data/pairs --per-topic 999 --samples 4 --max-new 100
done
echo "### PAIRS DONE $(date +%H:%M:%S)"
