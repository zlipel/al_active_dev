#!/bin/bash
# stage_acq_snapshots.sh — copy frozen training snapshots from production into a
# read-only staging area on the acquisition-diagnostic side.
#
# This is the ONLY script in the acquisition diagnostic that reads production
# (SCRATCH_AL). Run it once, review its output, then launch the cell grid
# (submit/run_acq_diag.sh); the cell jobs read only the staged copies and never
# open a production path. Production is never written.
#
# For each (model, iter) it copies features_gen{iter}.csv, labels_gen{iter}.csv
# and seq_gen{iter}.txt into:
#   ${SCRATCH_AL_ACQ}/<exp_name>/_snapshots/<model>/iteration_<iter>/
# and marks the staged copies read-only so nothing downstream can mutate them.
#
# Usage:
#   ./submit/stage_acq_snapshots.sh --model MPIPI --iters "1 6" [--exp_name NAME] [--force]
#
# Run directly on a login node (small copies); no sbatch needed.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/config/cluster.env"
source "${REPO_ROOT}/submit/acq_diag_lib.sh"

MODEL=""
ITERS=""
EXP_NAME="seminar_$(date +%Y%m%d)"
FORCE=false

usage() {
    cat <<EOF
Usage: $0 --model M --iters "N [N ...]" [--exp_name NAME] [--force]

Required:
  --model NAME       Force field (e.g. MPIPI)
  --iters "1 6"      Space-separated iteration snapshots to stage

Options:
  --exp_name NAME    Experiment group (default: seminar_YYYYMMDD)
  --force            Refuses replacement of an existing frozen snapshot; use a new name
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2"; shift ;;
        --iters)    ITERS="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --force)    FORCE=true ;;
        --help|-h)  usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; usage; }
[[ -z "$ITERS" ]] && { echo "Error: --iters is required"; usage; }
acq_validate_name "$MODEL" model
acq_validate_name "$EXP_NAME" exp_name
for ITER in $ITERS; do acq_validate_integer "$ITER" iteration; done
acq_activate_env

# Refuse to run if the acq roots are not truly separate from production.
assert_acq_roots_isolated

echo "Staging snapshots for ${MODEL} (exp: ${EXP_NAME})"
echo "  source (read-only): ${SCRATCH_AL}/${MODEL}/GENERATIONS/"
echo

for ITER in $ITERS; do
    SRC="${SCRATCH_AL}/${MODEL}/GENERATIONS/iteration_${ITER}"
    DST=$(acq_snapshot_dir "$EXP_NAME" "$MODEL" "$ITER")

    # Staging destination must resolve on the acq side (never a production dir).
    assert_under_acq_roots "$DST" "staging dir"

    FORCE_FLAG=()
    [[ "$FORCE" == true ]] && FORCE_FLAG=(--force)
    python "${REPO_ROOT}/submit/acq_diag_tools.py" stage \
        --source "$SRC" --destination "$DST" --iteration "$ITER" "${FORCE_FLAG[@]}"
done

echo
echo "Done. Staged snapshots are read-only under:"
echo "  ${SCRATCH_AL_ACQ}/${EXP_NAME}/_snapshots/${MODEL}/"
echo "Next: ./submit/run_acq_diag.sh --model ${MODEL} --exp_name ${EXP_NAME} ..."
