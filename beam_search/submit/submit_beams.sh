#!/bin/bash
# submit_beams.sh — submit a fresh beam search run for MODEL at ITER.
#
# Was: submit_beams_per_model.sh — used sed -i to mutate run_beams_per_model.sh on disk.
# Now: passes job-name/output/error as sbatch CLI flags (no on-disk mutation).

MODEL=$1
ITER=$2
NBINS=$3
KPERBIN=$4
LENGTH_CHANGES=$5
EXTEND_NO_FINISHED=${6:-false}
EXTRA_STEPS=${7:-0}
STAGNATION_PATIENCE=${8:-0}
STAGNATION_DELTA=${9:-0.0}

# ALPaths args — forwarded through to run_beams_mpi.py for checkpoint resolution
FRONT=${10:-upper}
EHVI_VARIANT=${11:-epsilon}
EXPLORATION_STRATEGY=${12:-kriging_believer}
TRANSFORM=${13:-yeoj}
MC_EHVI=${14:-false}

if [[ "$LENGTH_CHANGES" == "true" ]]; then
  LENGTH_TAG="length_changes"
else
  LENGTH_TAG="fixed_length"
fi

JOB_NAME="run_beams_${MODEL}_${LENGTH_TAG}"

sbatch \
  --job-name="${JOB_NAME}" \
  --output="${JOB_NAME}.out" \
  --error="${JOB_NAME}.err" \
  run_beams.sh \
    "$MODEL" "$ITER" "$NBINS" "$KPERBIN" \
    "$LENGTH_CHANGES" "$EXTEND_NO_FINISHED" "$EXTRA_STEPS" \
    "$STAGNATION_PATIENCE" "$STAGNATION_DELTA" \
    "$FRONT" "$EHVI_VARIANT" "$EXPLORATION_STRATEGY" "$TRANSFORM" "$MC_EHVI"
