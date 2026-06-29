#!/bin/bash
# run_acq_sweep.sh — submit one acquisition-function diagnostic job per
# (MODEL, EHVI_VARIANT, EXPLORATION_STRATEGY) tuple.
#
# Replaces the old runner.sh that sed-mutated #SBATCH headers; here job-name
# and log paths are set on the sbatch CLI flags instead.
#
# Usage:
#   ./submit/run_acq_sweep.sh MODEL EHVI_VARIANT EXPLORATION_STRATEGY
#
# Sweep with a bash loop:
#   for m in MPIPI CALVADOS HPS_URRY; do
#     for e in epsilon standard; do
#       ./submit/run_acq_sweep.sh "$m" "$e" kriging_believer
#     done
#   done

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 MODEL EHVI_VARIANT EXPLORATION_STRATEGY"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=$1
EHVI=$2
EXPLORE=$3

mkdir -p acq_logs

sbatch \
    --job-name="alp_${MODEL}_acq_test" \
    --output="acq_logs/acq_test_${MODEL}_${EHVI}_${EXPLORE}.out" \
    --error="acq_logs/acq_test_${MODEL}_${EHVI}_${EXPLORE}.err" \
    "${SCRIPT_DIR}/al_master_acq_test.sh" \
        --model "$MODEL" \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE"
