#!/bin/bash
#SBATCH --job-name=al_acq_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --output=acq_logs/al_acq_test.out
#SBATCH --error=acq_logs/al_acq_test.err
#
# Acquisition-function diagnostic: runs the AL pipeline at iter=0 with
# --acq_test on, against a SEPARATE scratch path so it doesn't clobber a
# production run. Designed to be invoked once per (model, ehvi, explore)
# combo, typically via submit/run_acq_sweep.sh.
#
# Usage:
#   sbatch submit/al_master_acq_test.sh --model MPIPI --ehvi_variant epsilon \
#                                       --exploration_strategy kriging_believer

set -eo pipefail

# Resolve repo root. SLURM rewrites BASH_SOURCE/$0 to a spool copy of the
# script; prefer SLURM_SUBMIT_DIR, then AL_ACTIVE_DEV env var, then the
# canonical install location.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/config/cluster.env" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${AL_ACTIVE_DEV:-}" && -f "${AL_ACTIVE_DEV}/config/cluster.env" ]]; then
    REPO_ROOT="${AL_ACTIVE_DEV}"
else
    REPO_ROOT="${HOME}/PROJECTS/al_active_dev"
fi
source "${REPO_ROOT}/config/cluster.env"

module purge
module load "${CONDA_MODULE}"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MODEL=""
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
FRONT="upper"
EPSILON_SCALE="1.0"
REF_POINT_MODE="frac"
MC_EHVI=true                  # default on for the diagnostic (matches old script)
NGEN=24
NCANDS=96
TRAIN_MODEL_TYPE="gpr_multitask"
TRANSFORM="yeoj"
OBJ1="exp_density"
OBJ2="diff"

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY, HPS_KR

Common sweep knobs:
  --ehvi_variant {epsilon,standard}            (default: epsilon)
  --exploration_strategy NAME                  (default: kriging_believer)
  --epsilon_scale F                            (default: 1.0)
  --ref_point_mode {frac,in_line,halfway}      (default: frac)
  --no-mc-ehvi                                 disable MC EHVI (default on)
  --front {upper,lower}                        (default: upper)

Power-user: pass-through with '--', e.g.
  $0 --model MPIPI -- --ref_point_frac 0.7
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --ehvi_variant) EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --epsilon_scale) EPSILON_SCALE="$2"; shift ;;
        --ref_point_mode) REF_POINT_MODE="$2"; shift ;;
        --no-mc-ehvi) MC_EHVI=false ;;
        --front) FRONT="$2"; shift ;;
        --ngen) NGEN="$2"; shift ;;
        --ncands) NCANDS="$2"; shift ;;
        --train_model_type) TRAIN_MODEL_TYPE="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --obj1) OBJ1="$2"; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --help|-h) usage ;;
        --) shift; EXTRA_FLAGS+=("$@"); break ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    usage
fi

CMD=(al-master
    --model "$MODEL"
    --iter 0
    --front "$FRONT"
    --ngen "$NGEN"
    --ncands "$NCANDS"
    --train_model_type "$TRAIN_MODEL_TYPE"
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --epsilon_scale "$EPSILON_SCALE"
    --ref_point_mode "$REF_POINT_MODE"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --base_path "$HOME_AL"
    --scratch_path "$SCRATCH_AL_ACQ"
    --db_path "$DB_PATH"
    --pessimism
    --acq_test
)
[[ "$MC_EHVI" == true ]] && CMD+=(--mc_ehvi)
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Don't relocate SLURM logs — acq tests rely on per-(model,ehvi,explore) names
# set by run_acq_sweep.sh on the sbatch CLI flags.

conda deactivate
