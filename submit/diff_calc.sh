#!/bin/bash
# diff_calc.sh — submit a diffusivity analysis job.
#
# Usage:
#   diff_calc.sh MODEL ITER [INNER OMP NSEQ]               # init (ITER=0) or AL iteration
#   diff_calc.sh MODEL --validation SCOPE [INNER OMP NSEQ] # post-hoc validation scope

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1

if [[ "${2:-}" == "--validation" ]]; then
    SCOPE="$3"
    SCOPE_SPEC="validation:$SCOPE"     # process_diff_sims.sh decodes this token
    TAG="val_${SCOPE,,}"
    INNER_JOBS=${4:-4}
    OMP_THREADS=${5:-4}
    NSEQ_JOBS=${6:-6}
else
    SCOPE_SPEC="$2"                     # integer iteration (0 = init)
    TAG="$2"
    INNER_JOBS=${3:-4}
    OMP_THREADS=${4:-4}
    NSEQ_JOBS=${5:-6}
fi

mkdir -p simlogs/DIFF
sbatch \
  --job-name="diff_${MODEL}_${TAG}" \
  --output="simlogs/DIFF/process_diff_${MODEL}_${TAG}.out" \
  --error="simlogs/DIFF/process_diff_${MODEL}_${TAG}.err" \
  "${SCRIPT_DIR}/process_diff_sims.sh" "$MODEL" "$SCOPE_SPEC" "$INNER_JOBS" "$OMP_THREADS" "$NSEQ_JOBS"
