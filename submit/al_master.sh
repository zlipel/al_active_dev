#!/bin/bash
#SBATCH --job-name=al_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1G
#SBATCH --time=05:59:00
#SBATCH --output=al_master.out
#SBATCH --error=al_master.err
#
# One sbatch per AL iteration. The master loops over child seq_ids internally
# (via cli/child.run_child), which itself fans out via multiprocessing across
# SLURM_CPUS_PER_TASK cores.
#
# Usage:
#   sbatch submit/al_master.sh --model MPIPI --iter 0 [override flags...]
#
# All flags pass through to al-master. See al_pipeline/core/config.py for the
# full list. Defaults below match the production config; override on the CLI.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/config/cluster.env"

module purge
module load "${CONDA_MODULE}"
conda activate "${CONDA_ENV}"

# Hard-cap threaded libraries so worker processes don't oversubscribe cores.
# (cli/child.py also sets these, but the master process needs them too.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Production defaults; CLI flags override.
MODEL=""
ITER=""
FRONT="upper"
NGEN=24
NCANDS=96
TRAIN_MODEL_TYPE="gpr_multitask"
TRANSFORM="yeoj"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
OBJ1="exp_density"
OBJ2="diff"
PESSIMISM=true

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M --iter N [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY, HPS_KR
  --iter N                  Active-learning iteration number (0, 1, 2, ...)

Common overrides (production defaults shown):
  --front {upper,lower}     Pareto front direction (default: upper)
  --ngen N                  Children per iteration (default: 24)
  --ncands N                GA candidates per child (default: 96)
  --transform {yeoj,log}    Label transform (default: yeoj)
  --ehvi_variant {epsilon,standard}
  --exploration_strategy {kriging_believer,similarity_penalty,constant_liar_min,...}
  --no-pessimism            Disable the on-by-default --pessimism flag
  --train_model_type {gpr_multitask,gpr_singletask,dnn}

Power-user: any other al-master flag can be passed through after '--', e.g.
  $0 --model MPIPI --iter 0 -- --ref_point_mode in_line --mc_ehvi
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --front) FRONT="$2"; shift ;;
        --ngen) NGEN="$2"; shift ;;
        --ncands) NCANDS="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --ehvi_variant) EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --train_model_type) TRAIN_MODEL_TYPE="$2"; shift ;;
        --obj1) OBJ1="$2"; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --no-pessimism) PESSIMISM=false ;;
        --help|-h) usage ;;
        --) shift; EXTRA_FLAGS+=("$@"); break ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL" || -z "$ITER" ]]; then
    echo "Error: --model and --iter are required"
    usage
fi

CMD=(al-master
    --model "$MODEL"
    --iter "$ITER"
    --front "$FRONT"
    --ngen "$NGEN"
    --ncands "$NCANDS"
    --train_model_type "$TRAIN_MODEL_TYPE"
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --base_path "$HOME_AL"
    --scratch_path "$SCRATCH_AL"
    --db_path "$DB_PATH"
)
[[ "$PESSIMISM" == true ]] && CMD+=(--pessimism)
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Move SLURM stdout/err into the iteration's log dir (matches old layout).
LOG_DEST="${HOME_AL}/${MODEL}/logs/iteration_${FRONT}_${ITER}"
mkdir -p "$LOG_DEST"
mv al_master.out "$LOG_DEST/al_master_iter${ITER}.out" 2>/dev/null || true
mv al_master.err "$LOG_DEST/al_master_iter${ITER}.err" 2>/dev/null || true

conda deactivate
