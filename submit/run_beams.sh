#!/bin/bash
#
# Runner for beam-search — fans `sbatch submit/init_beams.sh` out to one
# job per model.
#
# Not itself an sbatch entry. Runs on the login node. Each per-model
# submission gets `--job-name=init_beams_<MODEL>` so the `%x_%j.{out,err}`
# files written by init_beams.sh don't collide.
#
# All flags except `--models` and `--after-prep` are forwarded verbatim to
# init_beams.sh — mode / policy / profile / beam_width / etc. all belong
# there and shouldn't be re-parsed here.
#
# Also forwards sbatch-level flags via a `--sbatch_args <flag>` escape so
# per-run tuning (--ntasks, --time) doesn't require editing init_beams.sh:
#   ./submit/run_beams.sh --sbatch_args "--ntasks=5 --time=01:00:00" \
#       --mode benchmark --policy expert_tied --profile
#
# Usage:
#   ./submit/run_beams.sh --mode benchmark --policy expert_tied --profile
#   ./submit/run_beams.sh --models "HPS_URRY" --mode production --policy soft
#   ./submit/run_beams.sh --after-prep 123:124:125 --mode benchmark --policy expert_tied
#     # last form waits for the three prep jobs (colon-separated) to finish OK
#     # before starting each per-model beams run.

set -eo pipefail

if [[ -n "${AL_ACTIVE_DEV:-}" && -f "${AL_ACTIVE_DEV}/config/cluster.env" ]]; then
    REPO_ROOT="${AL_ACTIVE_DEV}"
elif [[ -f "$(pwd)/config/cluster.env" ]]; then
    REPO_ROOT="$(pwd)"
else
    REPO_ROOT="${HOME}/PROJECTS/al_active_dev"
fi

MODELS=("HPS_URRY" "MPIPI" "CALVADOS")
AFTER_PREP=""
SBATCH_FLAGS=""
FORWARD=()

usage() {
    cat <<EOF
Usage: $0 [--models "M1 M2 M3"] [--after-prep IDS] [--sbatch_args "FLAGS"]
          [flags forwarded to init_beams.sh]

Wrapper-only options:
  --models "M1 M2..."   Space-separated model list
                        (default: HPS_URRY MPIPI CALVADOS)
  --after-prep IDS      Colon-separated prep jobids to depend on
                        (chains via sbatch --dependency=afterok:IDS)
  --sbatch_args "FLAGS"      Extra sbatch-level flags — quoted string, e.g.
                        --sbatch_args "--ntasks=5 --time=01:00:00"

Everything else (--mode, --policy, --profile, --beam_width, etc.) is
forwarded verbatim to submit/init_beams.sh. Run
'submit/init_beams.sh --help' to see its options.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --models)     IFS=' ' read -r -a MODELS <<< "$2"; shift ;;
        --after-prep) AFTER_PREP="$2"; shift ;;
        --sbatch_args)     SBATCH_FLAGS="$2"; shift ;;
        --help|-h)    usage ;;
        *)            FORWARD+=("$1") ;;
    esac
    shift
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: --models resolved to empty list"
    usage
fi

# Build sbatch args as an array so quoted flags survive the expansion.
SBATCH_ARGS=(--parsable)
[[ -n "$SBATCH_FLAGS" ]] && SBATCH_ARGS+=($SBATCH_FLAGS)
[[ -n "$AFTER_PREP"  ]] && SBATCH_ARGS+=(--dependency="afterok:${AFTER_PREP}")

echo "Dispatching beams for models: ${MODELS[*]}"
[[ -n "$AFTER_PREP" ]] && echo "  chained after prep jobs: ${AFTER_PREP}"
JOB_IDS=()
for MODEL in "${MODELS[@]}"; do
    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        --job-name="init_beams_${MODEL}" \
        "${REPO_ROOT}/submit/init_beams.sh" \
        --model "$MODEL" \
        "${FORWARD[@]}")
    echo "  ${MODEL}  jobid=${JOB_ID}"
    JOB_IDS+=("$JOB_ID")
done

IFS=":" ; echo "BEAM_JOB_IDS=${JOB_IDS[*]}"
