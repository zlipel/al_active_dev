#!/bin/bash
# acq_diag_lib.sh — shared helpers for the acquisition diagnostic scripts.
# Sourced by stage_acq_snapshots.sh, acq_diag_cell.sh and run_acq_diag.sh AFTER
# config/cluster.env. Not executable on its own.
#
# The whole point of these helpers is one guarantee: the controlled experiment
# reads production data but can never write under the production roots
# (HOME_AL / SCRATCH_AL). Every write target is checked against those roots.

acq_activate_env() {
    module purge
    module load "${CONDA_MODULE}"
    conda activate "${CONDA_ENV}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

acq_validate_name() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        echo "FATAL: invalid ${2:-name}: $1"; exit 3;
    }
}

acq_validate_integer() {
    [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo "FATAL: invalid ${2:-integer}: $1"; exit 3;
    }
}

acq_cell_name() {
    local budget="full"
    [[ "$6" == true ]] && budget="pilot"
    echo "${1}_iter${2}_seed${3}_${4}_${5}_${budget}"
}

# Abort unless the acq roots are usable and not literally the production roots.
# NOTE: HOME_AL_ACQ is intentionally a dedicated subdir of HOME_AL
# (runs/ACQ_SWEEP), so nesting under HOME_AL is allowed and expected; only exact
# equality is a misconfiguration. SCRATCH_AL_ACQ is a separate root and must not
# nest under SCRATCH_AL. The write-time protection is assert_under_acq_roots.
assert_acq_roots_isolated() {
    python "${REPO_ROOT}/submit/acq_diag_tools.py" check-paths \
        --production-home "$HOME_AL" --production-scratch "$SCRATCH_AL" \
        --home-root "${HOME_AL_ACQ:-}" --scratch-root "${SCRATCH_AL_ACQ:-}"
}

# Positive guard: a write target MUST resolve under one of the acq roots and
# contain no '..' escape. This keeps every diagnostic write on the acq side
# (e.g. runs/ACQ_SWEEP/...) and out of production model dirs (runs/<MODEL>/).
assert_under_acq_roots() {
    local target="$1" label="${2:-path}"
    python "${REPO_ROOT}/submit/acq_diag_tools.py" check-paths \
        --production-home "$HOME_AL" --production-scratch "$SCRATCH_AL" \
        --home-root "$HOME_AL_ACQ" --scratch-root "$SCRATCH_AL_ACQ" \
        --target "$target" || { echo "FATAL: invalid ${label}: $target"; exit 3; }
}

# Canonical staging location for a frozen (model, iter) snapshot on the acq side.
# echo the directory path; callers create/read it.
acq_snapshot_dir() {
    local exp_name="$1" model="$2" iter="$3"
    echo "${SCRATCH_AL_ACQ}/${exp_name}/_snapshots/${model}/iteration_${iter}"
}

# The three files that make up a frozen training snapshot for iteration N.
acq_snapshot_files() {
    local iter="$1"
    echo "features_gen${iter}.csv" "labels_gen${iter}.csv" "seq_gen${iter}.txt"
}
