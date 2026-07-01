#!/bin/bash
# run_moe_diagnostic_sweep.sh — submit one MoE retrospective job per
# (MODEL, FRONT) tuple.
#
# Each submission runs `submit/moe_diagnostic.sh` for that model/front. Jobs
# are independent — no ordering required between them.
#
# Usage:
#   ./submit/run_moe_diagnostic_sweep.sh MODEL FRONT [--n_iters N] [--k_pick K] [extra flags...]
#
# Sweep across all four models and both fronts (upper + lower):
#   for m in MPIPI CALVADOS HPS_URRY HPS_KR; do
#     for f in upper lower; do
#       ./submit/run_moe_diagnostic_sweep.sh "$m" "$f"
#     done
#   done

set -eo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL FRONT [--n_iters N] [--k_pick K] [extra flags...]"
    echo "  MODEL: MPIPI | CALVADOS | HPS_URRY | HPS_KR"
    echo "  FRONT: upper | lower"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=$1
FRONT=$2
shift 2

# Remaining args pass through to moe_diagnostic.sh (which itself forwards
# unknown flags into the moe_diagnostic CLI).
EXTRA_ARGS=("$@")

mkdir -p diagnostic_logs

for s in 1 3 5 7; do

    sbatch \
        --job-name="moe_diag_${MODEL}_${FRONT}" \
        --output="diagnostic_logs/moe_diag_${MODEL}_${FRONT}_StIter_${s}.out" \
        --error="diagnostic_logs/moe_diag_${MODEL}_${FRONT}_StIter_${s}.err" \
        "${SCRIPT_DIR}/moe_diagnostic.sh" \
            --model "$MODEL" \
            --front "$FRONT" \
            --start_iter "$s" \
            "${EXTRA_ARGS[@]}"
