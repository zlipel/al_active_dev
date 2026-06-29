#!/bin/bash
# eos_calc.sh — submit an EOS analysis job for MODEL at ITER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1
NBOOT=$2
ITER=$3

mkdir -p simlogs/EOS
sbatch \
  --job-name="eos_${MODEL}_${ITER}" \
  --output="simlogs/EOS/process_eos_${MODEL}_${ITER}.out" \
  --error="simlogs/EOS/process_eos_${MODEL}_${ITER}.err" \
  "${SCRIPT_DIR}/process_eos_sims.sh" "$MODEL" "$NBOOT" "$ITER"
