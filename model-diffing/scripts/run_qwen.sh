#!/bin/bash
set -e
export HF_HOME=/Users/skyenygaard/Programming/spar-portfolio/activation-introspection/hf_cache
PY=../activation-introspection/.venv/bin/python
for a in bad-medical-advice risky-financial-advice extreme-sports; do
  echo "### $a $(date +%H:%M:%S)"
  $PY scripts/collect.py \
    --base unsloth/Qwen2.5-0.5B-Instruct \
    --adapter "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_$a" \
    --tag "qwen05b_${a}" --per-topic 30 --samples 4 --max-new 100
done
echo "### QWEN DONE $(date +%H:%M:%S)"
