#!/bin/bash
#
# Runner for endpoint prep — fans `sbatch submit/prepare_endpoints.sh` out
# to one job per model.
#
# Not itself an sbatch entry. Runs on the login node. Each per-model
# submission gets `--job-name=prep_endpoints_<MODEL>` so the
# `%x_%j.{out,err}` files written by prepare_endpoints.sh don't collide
# across models running in parallel.
#
# All flags except `--models` are forwarded verbatim to prepare_endpoints.sh
# — mode/thresholds/grid/counts all belong there and shouldn't be
# re-parsed here.
#
# Usage:
#   ./submit/run_prep_endpoints.sh --mode benchmark
#   ./submit/run_prep_endpoints.sh --mode production --frac_ps 0.9 --frac_nonps 0.75
#   ./submit/run_prep_endpoints.sh --models "HPS_URRY MPIPI" --mode benchmark

set -eo pipefail

if [[ -n "${AL_ACTIVE_DEV:-}" && -f "${AL_ACTIVE_DEV}/config/cluster.env" ]]; then
    REPO_ROOT="${AL_ACTIVE_DEV}"
elif [[ -f "$(pwd)/config/cluster.env" ]]; then
    REPO_ROOT="$(pwd)"
else
    REPO_ROOT="${HOME}/PROJECTS/al_active_dev"
fi

MODELS=("HPS_URRY" "MPIPI" "CALVADOS")
FORWARD=()

usage() {
    cat <<EOF
Usage: $0 [--models "M1 M2 M3"] [flags forwarded to prepare_endpoints.sh]

Wrapper-only options:
  --models "M1 M2..."   Space-separated model list
                        (default: HPS_URRY MPIPI CALVADOS)

Everything else (--mode, --n_ps, --frac_ps, --thresh_lower, etc.) is
forwarded verbatim to submit/prepare_endpoints.sh. Run
'submit/prepare_endpoints.sh --help' to see its options.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --models) IFS=' ' read -r -a MODELS <<< "$2"; shift ;;
        --help|-h) usage ;;
        *) FORWARD+=("$1") ;;
    esac
    shift
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: --models resolved to empty list"
    usage
fi

echo "Dispatching prepare_endpoints for models: ${MODELS[*]}"
JOB_IDS=()
for MODEL in "${MODELS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="prep_endpoints_${MODEL}" \
        "${REPO_ROOT}/submit/prepare_endpoints.sh" \
        --model "$MODEL" \
        "${FORWARD[@]}")
    echo "  ${MODEL}  jobid=${JOB_ID}"
    JOB_IDS+=("$JOB_ID")
done

# Colon-joined jobids so `run_beams.sh --after-prep <IDS>` can chain via
# `sbatch --dependency=afterok:<IDS>`. Prefix stripped by the beams wrapper.
IFS=":" ; echo "PREP_JOB_IDS=${JOB_IDS[*]}"
