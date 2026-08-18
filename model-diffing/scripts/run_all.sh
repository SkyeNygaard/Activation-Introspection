#!/bin/bash
set -e
export HF_HOME=/Users/skyenygaard/Programming/spar-portfolio/activation-introspection/hf_cache
PY=../activation-introspection/.venv/bin/python
for a in bad-medical-advice risky-financial-advice extreme-sports; do
  echo "### $a $(date +%H:%M:%S)"
  $PY scripts/collect.py \
    --base unsloth/Llama-3.2-1B-Instruct \
    --adapter "ModelOrganismsForEM/Llama-3.2-1B-Instruct_$a" \
    --tag "llama1b_${a}" --per-topic 30 --samples 4 --max-new 100
done
echo "### ALL DONE $(date +%H:%M:%S)"
