#!/bin/bash
# submit_phase_separated.sh — submit a phase-separated endpoint append + resume.
#
# Was: submit_phase_separated_per_model.sh — used sed -i to mutate job script on disk.
# Now: passes job-name/output/error as sbatch CLI flags (no on-disk mutation).

MODEL=${1:-CALVADOS}
ITER=${2:-10}
LENGTH_CHANGES=${3:-false}
EXTEND_NO_FINISHED=${4:-false}
EXTRA_STEPS=${5:-0}
STAGNATION_PATIENCE=${6:-0}
STAGNATION_DELTA=${7:-0.0}

# ALPaths args
FRONT=${8:-upper}
EHVI_VARIANT=${9:-epsilon}
EXPLORATION_STRATEGY=${10:-kriging_believer}
TRANSFORM=${11:-yeoj}
MC_EHVI=${12:-false}

if [[ "$LENGTH_CHANGES" == "true" ]]; then
  LENGTH_TAG="length_changes"
else
  LENGTH_TAG="fixed_length"
fi

JOB_NAME="resume_beams_PS_${MODEL}_${LENGTH_TAG}"

sbatch \
  --job-name="${JOB_NAME}" \
  --output="${JOB_NAME}.out" \
  --error="${JOB_NAME}.err" \
  resume_phase_separated.sh \
    "$MODEL" "$ITER" "$LENGTH_CHANGES" \
    "$EXTEND_NO_FINISHED" "$EXTRA_STEPS" \
    "$STAGNATION_PATIENCE" "$STAGNATION_DELTA" \
    "$FRONT" "$EHVI_VARIANT" "$EXPLORATION_STRATEGY" "$TRANSFORM" "$MC_EHVI"
