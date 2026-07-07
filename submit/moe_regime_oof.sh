#!/bin/bash
#SBATCH --job-name=moe_oof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=moe_oof.out
#SBATCH --error=moe_oof.err
#
# MoE regime-OOF diagnostic + final production-model training for ONE model
# / iteration.
#
# Runs stratified k-fold OOF on the labeled data at ${ITER} and reports
# metrics on the full six-predictor set (global, ps_expert, nonps_expert,
# moe_soft, moe_hard, ps_guarded) across physical + z spaces and the PS /
# nonPS / density-quartile / p_ps-bin splits.
#
# Optionally (default: on) trains the FINAL production models the beam
# search will consume:
#   - PS + nonPS + calibrated RF gate via the standard MoE training
#     path (`train_moe_from_config`) — writes to MODELS/MOE_{PS,NONPS,RF}_*.
#   - Global multitask GPR via the standard AL training path
#     (`train_from_config` with train_model_type='gpr_multitask') — writes
#     to MODELS/GPR_iter*.pt.
#
# Reads training data from
#   ${SCRATCH_AL}/<MODEL>/GENERATIONS/iteration_${ITER}/
# (via --scratch_path). Writes diagnostic outputs to
#   ${HOME_AL}/<MODEL>/DIAGNOSTIC/regime_oof_*
# and model artifacts under MODELS/ per the standard AL layout.
#
# Usage:
#   sbatch submit/moe_regime_oof.sh --model HPS_URRY --iter 10
#   sbatch submit/moe_regime_oof.sh --model CALVADOS --iter 10 --n_folds 5 --skip_final
#   sbatch submit/moe_regime_oof.sh --model MPIPI --iter 10 --skip_oof

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

# Defaults match production AL training params. Override via CLI.
MODEL=""
ITER=""
FRONT="upper"
TRANSFORM="yeoj"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
OBJ1="exp_density"
OBJ2="diff"
NGEN=24
N_FOLDS=5
PS_THRESHOLD=0.5
EPOCHS=1000
PATIENCE=5
K_FOLDS=5
LEARNING_RATE=0.1
SKIP_OOF="false"
SKIP_FINAL="false"

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M --iter N [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY, HPS_KR
  --iter N                  Iteration to evaluate (typically the final iter)

Common overrides (production defaults shown):
  --front {upper,lower}     ALConfig requires it but the OOF diagnostic is
                            front-agnostic (default: upper)
  --transform {yeoj,log}    Label transform (default: yeoj)
  --n_folds N               Stratified k-fold splits (default: 5)
  --ps_threshold T          Default gate threshold for moe_hard/ps_guarded
                            (metrics also sweep {0.15, 0.30, 0.50, 0.70};
                            default: 0.5)
  --skip_oof                Skip OOF diagnostic (only train final models)
  --skip_final              Skip final training (only run OOF)
  --ngen N                  Batch size used by the original AL run (default: 24)

Any other flag is forwarded to moe_regime_oof unchanged.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --front) FRONT="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --ehvi_variant) EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --obj1) OBJ1="$2"; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --ngen) NGEN="$2"; shift ;;
        --n_folds) N_FOLDS="$2"; shift ;;
        --ps_threshold) PS_THRESHOLD="$2"; shift ;;
        --skip_oof) SKIP_OOF="true" ;;
        --skip_final) SKIP_FINAL="true" ;;
        --help|-h) usage ;;
        --) shift; EXTRA_FLAGS+=("$@"); break ;;
        *) EXTRA_FLAGS+=("$1") ;;
    esac
    shift
done

if [[ -z "$MODEL" || -z "$ITER" ]]; then
    echo "Error: --model and --iter are required"
    usage
fi

CMD=(python -m al_pipeline.cli.moe_regime_oof
    --model "$MODEL"
    --iter "$ITER"
    --front "$FRONT"
    --train_model_type moe
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --ngen "$NGEN"
    --n_folds "$N_FOLDS"
    --ps_threshold "$PS_THRESHOLD"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --k_folds "$K_FOLDS"
    --learning_rate "$LEARNING_RATE"
    --base_path "$HOME_AL"
    --scratch_path "$SCRATCH_AL"
    --db_path "$DB_PATH"
)
if [[ "$SKIP_OOF" == "true" ]]; then
    CMD+=(--skip_oof)
fi
if [[ "$SKIP_FINAL" == "true" ]]; then
    CMD+=(--skip_final)
fi
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Move SLURM stdout/err into the diagnostic dir alongside the outputs.
LOG_DEST="${HOME_AL}/${MODEL}/DIAGNOSTIC"
mkdir -p "$LOG_DEST"
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/moe_oof.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/moe_oof.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/moe_oof_iter${ITER}.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/moe_oof_iter${ITER}.err"

conda deactivate
