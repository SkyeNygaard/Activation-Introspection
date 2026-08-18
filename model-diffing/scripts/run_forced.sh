#!/bin/bash
set -e
export HF_HOME=/Users/skyenygaard/Programming/spar-portfolio/activation-introspection/hf_cache
PY=../activation-introspection/.venv/bin/python
for a in bad-medical-advice risky-financial-advice extreme-sports; do
  echo "### $a $(date +%H:%M:%S)"
  $PY scripts/forced.py --tag "llama1b_$a" --adapter "ModelOrganismsForEM/Llama-3.2-1B-Instruct_$a" --per-question 2
done
echo "### FORCED DONE $(date +%H:%M:%S)"
