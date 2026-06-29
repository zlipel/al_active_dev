#!/bin/bash
# eos_calc.sh — submit an EOS analysis job for MODEL at ITER.

MODEL=$1
NBOOT=$2
ITER=$3

sbatch \
  --job-name="eos_${MODEL}_${ITER}" \
  --output="simlogs/EOS/process_eos_${MODEL}_${ITER}.out" \
  --error="simlogs/EOS/process_eos_${MODEL}_${ITER}.err" \
  process_eos_sims.sh "$MODEL" "$NBOOT" "$ITER"
