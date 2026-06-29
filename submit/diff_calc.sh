#!/bin/bash
# diff_calc.sh — submit a diffusivity analysis job for MODEL at ITER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1
ITER=$2
INNER_JOBS=${3:-4}
OMP_THREADS=${4:-4}
NSEQ_JOBS=${5:-6}

mkdir -p simlogs/DIFF
sbatch \
  --job-name="diff_${MODEL}_${ITER}" \
  --output="simlogs/DIFF/process_diff_${MODEL}_${ITER}.out" \
  --error="simlogs/DIFF/process_diff_${MODEL}_${ITER}.err" \
  "${SCRIPT_DIR}/process_diff_sims.sh" "$MODEL" "$ITER" "$INNER_JOBS" "$OMP_THREADS" "$NSEQ_JOBS"
