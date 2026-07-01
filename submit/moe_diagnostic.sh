#!/bin/bash
#SBATCH --job-name=moe_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --output=moe_diag.out
#SBATCH --error=moe_diag.err
#
# MoE retrospective diagnostic for ONE model / front.
#
# Reads the completed 10-round global-AL artifacts from
#   ${SCRATCH_AL}/<MODEL>/GENERATIONS/iteration_*/
# (via --scratch_path; this is where the AL loop's features/labels actually
# live — HOME_AL only holds outputs). Refits MoE + global surrogates per iter
# on the same training slice, ranks each iter's real children by EHVI under
# each surrogate, and rolls cumulative HV forward under a counterfactual
# "half-batch" pick (top-K by EHVI where K = ngen // 2).
#
# Output: ${HOME_AL}/<MODEL>/DIAGNOSTIC/  (via --base_path)
#   retrospective_summary.csv
#   retrospective_trajectory.json
#   retrospective_hv.png
#
# Not resource-intensive — 4 cores, ~1h wall for one model. Runs in a
# single job (no LAMMPS, small in-memory refits).
#
# Usage:
#   sbatch submit/moe_diagnostic.sh --model MPIPI --front upper
#   sbatch submit/moe_diagnostic.sh --model CALVADOS --front upper --n_iters 8 --k_pick 12

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

# Cap threaded libs — surrogate refits are single-threaded here, no benefit
# from wider BLAS and it messes with the timing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Defaults match production AL loop; override via CLI.
MODEL=""
FRONT="upper"
N_ITERS=10
K_PICK=""
TRANSFORM="yeoj"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
OBJ1="exp_density"
OBJ2="diff"
NGEN=24
MOE_POLICY="soft"
# Production-shape training params (matches al_master.sh). The retrospective
# refits surrogates per iter; toy values here give toy rankings.
EPOCHS=1000
PATIENCE=5
K_FOLDS=5
LEARNING_RATE=0.1
PESSIMISM_START_ITER=6

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY, HPS_KR

Common overrides (production defaults shown):
  --front {upper,lower}     Pareto front direction (default: upper)
  --n_iters N               Number of completed iters to walk (default: 10)
  --k_pick N                Top-K children per iter (default: ngen // 2 = 12)
  --transform {yeoj,log}    Label transform (default: yeoj)
  --moe_policy {soft,hard}  MoE blending policy (default: soft)
  --ngen N                  Batch size used by the original AL run (default: 24)

Any other flag is forwarded to moe_diagnostic unchanged.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --front) FRONT="$2"; shift ;;
        --n_iters) N_ITERS="$2"; shift ;;
        --k_pick) K_PICK="$2"; shift ;;
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

CMD=(python -m al_pipeline.cli.moe_diagnostic
    --n_iters "$N_ITERS"
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
    --pessimism_start_iter "$PESSIMISM_START_ITER"
    --base_path "$HOME_AL"
    --scratch_path "$SCRATCH_AL"
    --db_path "$DB_PATH"
)
# NOTE: no --runs_root: the CLI defaults to cfg.scratch_path (= $SCRATCH_AL),
# which is where the AL loop's features/labels/seqs actually live. Only set
# --runs_root explicitly if you're pointing at an archived copy elsewhere.
[[ -n "$K_PICK" ]] && CMD+=(--k_pick "$K_PICK")
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Move SLURM stdout/err into the diagnostic dir alongside the outputs.
LOG_DEST="${HOME_AL}/${MODEL}/DIAGNOSTIC"
mkdir -p "$LOG_DEST"
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/moe_diag.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/moe_diag.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/moe_diagnostic.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/moe_diagnostic.err"

conda deactivate
