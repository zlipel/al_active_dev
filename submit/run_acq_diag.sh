#!/bin/bash
# run_acq_diag.sh — submit the acquisition-diagnostic grid: one cell job per
# (iter, method, seed). Verifies that snapshots are already staged (acq side)
# and never touches production itself.
#
# The experiment: on MPIPI, upper front, global multitask GP, compare ordinary
# kriging-believer (kb) vs pessimistic kriging-believer (pkb) at an early snapshot
# (iter 1 -> round 2) and a late one (iter 6 -> round 7), across paired GA seeds.
# The question is whether pessimism helps more in the later round than the early
# one.
#
# Defaults reproduce the "start with one paired seed" step: iters {1,6} x
# methods {kb,pkb} x seeds {12345} = 4 jobs. Expand seeds to three for 12 jobs;
# add 'independent' for the no-within-batch-update control.
#
# Usage:
#   # one paired seed (4 jobs)
#   ./submit/run_acq_diag.sh --model MPIPI --exp_name seminar_20260923
#   # full three-seed design (12 jobs)
#   ./submit/run_acq_diag.sh --model MPIPI --exp_name seminar_20260923 \
#       --seeds "12345 23456 34567"
#   # add the independent control (18 jobs)
#   ./submit/run_acq_diag.sh --model MPIPI --exp_name seminar_20260923 \
#       --seeds "12345 23456 34567" --methods "kb pkb independent"
#   # tiny plumbing check first
#   ./submit/run_acq_diag.sh --model MPIPI --exp_name pilot_20260923 --pilot --dry_run

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/config/cluster.env"
source "${REPO_ROOT}/submit/acq_diag_lib.sh"

MODEL="MPIPI"
FRONT="upper"
ITERS="1 6"
METHODS="kb pkb"
SEEDS="12345"
EXP_NAME="seminar_$(date +%Y%m%d)"
PILOT=false
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $0 [--model M] [--exp_name NAME] [options]

Options (defaults shown):
  --model NAME         MPIPI
  --front {upper,lower} upper
  --iters "1 6"        snapshots to test
  --methods "kb pkb"   subset of {kb, pkb, independent}
  --seeds "12345"      GA seeds; use "12345 23456 34567" for the full design
  --exp_name NAME      seminar_YYYYMMDD
  --pilot              tiny budget per cell (plumbing check)
  --dry_run            print the grid without submitting

Snapshots must be staged first:
  ./submit/stage_acq_snapshots.sh --model M --iters "1 6" --exp_name NAME
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2"; shift ;;
        --front)    FRONT="$2"; shift ;;
        --iters)    ITERS="$2"; shift ;;
        --methods)  METHODS="$2"; shift ;;
        --seeds)    SEEDS="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --pilot)    PILOT=true ;;
        --dry_run)  DRY_RUN=true ;;
        --help|-h)  usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

acq_validate_name "$MODEL" model
acq_validate_name "$EXP_NAME" exp_name
[[ "$FRONT" == upper || "$FRONT" == lower ]] || { echo "Invalid front: $FRONT"; exit 2; }
[[ -n "$ITERS" && -n "$METHODS" && -n "$SEEDS" ]] || { echo "Empty grid axis"; exit 2; }
for ITER in $ITERS; do acq_validate_integer "$ITER" iteration; done
for SEED in $SEEDS; do acq_validate_integer "$SEED" seed; done
for METHOD in $METHODS; do
    [[ "$METHOD" == kb || "$METHOD" == pkb || "$METHOD" == independent ]] || {
        echo "Invalid method: $METHOD"; exit 2;
    }
done
acq_activate_env
assert_acq_roots_isolated

# Verify every needed snapshot is staged before submitting anything.
MISSING=false
for ITER in $ITERS; do
    STAGE_DIR=$(acq_snapshot_dir "$EXP_NAME" "$MODEL" "$ITER")
    assert_under_acq_roots "$STAGE_DIR" "snapshot"
    for name in $(acq_snapshot_files "$ITER"); do
        if [[ ! -f "${STAGE_DIR}/${name}" ]]; then
            echo "Missing staged snapshot: ${STAGE_DIR}/${name}"
            MISSING=true
        fi
    done
    if [[ -d "$STAGE_DIR" ]]; then
        python "${REPO_ROOT}/submit/acq_diag_tools.py" verify --directory "$STAGE_DIR" --iteration "$ITER" || MISSING=true
    fi
done
if [[ "$MISSING" == true ]]; then
    echo
    echo "Stage snapshots first:"
    echo "  ./submit/stage_acq_snapshots.sh --model ${MODEL} --iters \"${ITERS}\" --exp_name ${EXP_NAME}"
    exit 1
fi

# Per-cell wall time and pilot flag.
if [[ "$PILOT" == true ]]; then TIME="00:30:00"; PILOT_FLAG=(--pilot); else TIME="03:59:00"; PILOT_FLAG=(); fi

mkdir -p "${REPO_ROOT}/acq_logs"

N=0
echo "Acquisition diagnostic grid (exp: ${EXP_NAME}, model: ${MODEL}, front: ${FRONT}, pilot: ${PILOT})"
for ITER in $ITERS; do
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            CELL=$(acq_cell_name "$MODEL" "$ITER" "$SEED" "$METHOD" "$FRONT" "$PILOT")
            N=$((N + 1))
            if [[ "$DRY_RUN" == true ]]; then
                echo "  [dry-run] ${CELL}"
                continue
            fi
            sbatch \
                --chdir="${REPO_ROOT}" \
                --job-name="acqd_${CELL}" \
                --time="${TIME}" \
                --output="acq_logs/acqd_${CELL}_%j.out" \
                --error="acq_logs/acqd_${CELL}_%j.err" \
                "${REPO_ROOT}/submit/acq_diag_cell.sh" \
                    --model "$MODEL" \
                    --iter "$ITER" \
                    --seed "$SEED" \
                    --method "$METHOD" \
                    --front "$FRONT" \
                    --exp_name "$EXP_NAME" \
                    "${PILOT_FLAG[@]}"
            echo "  submitted ${CELL}"
        done
    done
done

echo
if [[ "$DRY_RUN" == true ]]; then echo "${N} cell(s) would be queued."; else echo "${N} cell(s) queued."; fi
BUDGET_NAME=full
[[ "$PILOT" == true ]] && BUDGET_NAME=pilot
[[ "$DRY_RUN" == true ]] || echo "Analyze when done: python utils/acq_diag_report.py --exp_name ${EXP_NAME} --model ${MODEL} --front ${FRONT} --budget ${BUDGET_NAME} --home_root ${HOME_AL_ACQ}"
