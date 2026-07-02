#!/bin/bash
#SBATCH --job-name=moe_fwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --output=moe_fwd.out
#SBATCH --error=moe_fwd.err
#
# MoE forward (generation-forward) diagnostic for ONE model.
#
# Trains ONE surrogate set per iter on gens 0..N-1 and evaluates it on gen N.
# Covers BOTH fronts in a single run — the model is front-agnostic; per-row
# `front_type` (upper/lower) is inferred from the gen-N pool's row order.
# --front is required by ALConfig but ignored by the forward evaluation.
#
# Reports the six operationally-deployed predictors — `global`, `moe_soft`,
# and `moe_hard_t{015,030,050,070}` (hard-gate at p_ps threshold 0.15..0.70).
# Raw expert predictions are NOT reported: evaluating one expert on rows the
# gate would never route to it has no operational meaning. The MoE RF gate
# is calibrated via CalibratedClassifierCV (see cfg.moe_calibration_method).
#
# Reads completed AL artifacts from
#   ${SCRATCH_AL}/<MODEL>/GENERATIONS/iteration_*/
# (via --scratch_path). Writes four CSVs + one plot to
#   ${HOME_AL}/<MODEL>/DIAGNOSTIC/
# with a `_start{N}` suffix so different --start_iter values coexist.
#
# Usage:
#   sbatch submit/moe_forward_diagnostic.sh --model MPIPI
#   sbatch submit/moe_forward_diagnostic.sh --model HPS_URRY --start_iter 3

set -eo pipefail

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

# Defaults match production AL loop; override via CLI.
MODEL=""
FRONT="upper"
N_ITERS=10
START_ITER=1
TRANSFORM="yeoj"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
OBJ1="exp_density"
OBJ2="diff"
NGEN=24
MOE_POLICY="soft"
# Production-shape training params. Toy values give toy accuracy metrics.
EPOCHS=1000
PATIENCE=5
K_FOLDS=5
LEARNING_RATE=0.1

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY, HPS_KR

Common overrides (production defaults shown):
  --front {upper,lower}     Pareto front direction (default: upper)
  --n_iters N               Number of completed iters to walk (default: 10)
  --start_iter N            First iter to evaluate (default: 1)
  --transform {yeoj,log}    Label transform (default: yeoj)
  --moe_policy {soft,hard}  MoE blending policy (default: soft)
  --ngen N                  Batch size used by the original AL run (default: 24)

Any other flag is forwarded to moe_forward_diagnostic unchanged. Output
filenames get a _start{N} suffix so sweeps land side-by-side.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --front) FRONT="$2"; shift ;;
        --n_iters) N_ITERS="$2"; shift ;;
        --start_iter) START_ITER="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --ehvi_variant) EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --obj1) OBJ1="$2"; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --ngen) NGEN="$2"; shift ;;
        --moe_policy) MOE_POLICY="$2"; shift ;;
        --help|-h) usage ;;
        --) shift; EXTRA_FLAGS+=("$@"); break ;;
        *) EXTRA_FLAGS+=("$1") ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    usage
fi

CMD=(python -m al_pipeline.cli.moe_forward_diagnostic
    --n_iters "$N_ITERS"
    --start_iter "$START_ITER"
    --model "$MODEL"
    --iter 0
    --front "$FRONT"
    --train_model_type moe
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --ngen "$NGEN"
    --moe_policy "$MOE_POLICY"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --k_folds "$K_FOLDS"
    --learning_rate "$LEARNING_RATE"
    --base_path "$HOME_AL"
    --scratch_path "$SCRATCH_AL"
    --db_path "$DB_PATH"
)
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Move SLURM stdout/err into the diagnostic dir alongside the outputs.
LOG_DEST="${HOME_AL}/${MODEL}/DIAGNOSTIC"
mkdir -p "$LOG_DEST"
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/moe_fwd.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/moe_fwd.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/moe_fwd_start${START_ITER}.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/moe_fwd_start${START_ITER}.err"

conda deactivate
