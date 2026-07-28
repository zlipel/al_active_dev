#!/bin/bash
# eos_calc.sh — submit an EOS analysis job.
#
# Usage:
#   eos_calc.sh MODEL NBOOT ITER                 # init (ITER=0) or AL iteration
#   eos_calc.sh MODEL NBOOT --validation SCOPE   # post-hoc validation scope

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1
NBOOT=$2

if [[ "${3:-}" == "--validation" ]]; then
    SCOPE="$4"
    SCOPE_SPEC="validation:$SCOPE"     # process_eos_sims.sh decodes this token
    TAG="val_${SCOPE,,}"
else
    SCOPE_SPEC="$3"                     # integer iteration (0 = init)
    TAG="$3"
fi

mkdir -p simlogs/EOS
sbatch \
  --job-name="eos_${MODEL}_${TAG}" \
  --output="simlogs/EOS/process_eos_${MODEL}_${TAG}.out" \
  --error="simlogs/EOS/process_eos_${MODEL}_${TAG}.err" \
  "${SCRIPT_DIR}/process_eos_sims.sh" "$MODEL" "$NBOOT" "$SCOPE_SPEC"
