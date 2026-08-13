#!/bin/bash
#SBATCH --job-name=forward_round
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1G
#SBATCH --time=03:59:00
#SBATCH --output=acq_logs/forward_round_%j.out
#SBATCH --error=acq_logs/forward_round_%j.err
#
# Forward-convergence round: re-run the next AL acquisition under one surrogate
# mode in an isolated tree. See README (Forward-convergence round) and --help.
# Run the three modes sequentially.

set -eo pipefail

# Resolve repo root (same pattern as al_master_acq_test.sh).
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

# numba + gpytorch scale best single-threaded here; keep the GA workers from
# oversubscribing (num_workers = min(ncands, SLURM_CPUS_PER_TASK)).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MODEL=""
FRONT="upper"
MODE=""
NGEN=8
ITER=10
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
TRANSFORM="yeoj"
OBJ1="exp_density"
OBJ2="diff"
SEED_DIR=""                                   # default filled in after MODEL is known
FWD_ROOT="${SCRATCH_AL_ACQ}/FORWARD_ITER11"   # isolated throwaway tree

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M --mode {global,soft,hard} [options]

Required:
  --model NAME              Force field: MPIPI, CALVADOS, HPS_URRY
  --mode  {global,soft,hard}  Surrogate policy (-> --run_tag)

Options:
  --front {upper,lower}     (default: upper)
  --ngen  K                 kriging-believer picks / batch size (default: 8)
  --iter  N                 iteration to re-run (default: 10 -> proposes gen-11)
  --seed_dir DIR            curated iteration_{iter} CSVs source
                            (default: \$SCRATCH_AL/<MODEL>/GENERATIONS/iteration_{iter})
  --fwd_root DIR            isolated tree root (default: \$SCRATCH_AL_ACQ/FORWARD_ITER11)

Any other flag is forwarded to al-master unchanged ('--' also works).
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --front) FRONT="$2"; shift ;;
        --ngen) NGEN="$2"; shift ;;
        --iter) ITER="$2"; shift ;;
        --seed_dir) SEED_DIR="$2"; shift ;;
        --fwd_root) FWD_ROOT="$2"; shift ;;
        --ehvi_variant) EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --obj1) OBJ1="$2"; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --help|-h) usage ;;
        --) shift; EXTRA_FLAGS+=("$@"); break ;;
        *) EXTRA_FLAGS+=("$1") ;;
    esac
    shift
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; usage; }
[[ -z "$MODE"  ]] && { echo "Error: --mode is required";  usage; }

# Map mode -> surrogate flags + run_tag.
case "$MODE" in
    global) SURR=(--train_model_type gpr_multitask) ;;
    soft)   SURR=(--train_model_type moe --moe_policy soft) ;;
    hard)   SURR=(--train_model_type moe --moe_policy hard) ;;
    *) echo "Error: --mode must be global|soft|hard, got '$MODE'"; usage ;;
esac
RUN_TAG="$MODE"

# Isolated scratch + base (production HOME_AL / SCRATCH_AL untouched).
FWD_SCRATCH="${FWD_ROOT}/scratch"
FWD_HOME="${FWD_ROOT}/home"

# Seed source (curated gen-{ITER} CSVs) and destination in the isolated tree.
[[ -z "$SEED_DIR" ]] && SEED_DIR="${SCRATCH_AL}/${MODEL}/GENERATIONS/iteration_${ITER}"
DEST="${FWD_SCRATCH}/${MODEL}/GENERATIONS/iteration_${ITER}"
mkdir -p "$DEST"

echo "Seeding curated gen-${ITER} CSVs: ${SEED_DIR} -> ${DEST}"
for f in "features_gen${ITER}.csv" "labels_gen${ITER}.csv" "seq_gen${ITER}.txt"; do
    if [[ ! -f "${SEED_DIR}/${f}" ]]; then
        echo "Error: missing seed file ${SEED_DIR}/${f}"
        echo "       point --seed_dir at your curated iteration_${ITER} directory."
        exit 1
    fi
    cp -f "${SEED_DIR}/${f}" "${DEST}/${f}"
done

CMD=(al-master
    --model "$MODEL"
    --iter "$ITER"
    --front "$FRONT"
    --ngen "$NGEN"
    "${SURR[@]}"
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --base_path "$FWD_HOME"
    --scratch_path "$FWD_SCRATCH"
    --db_path "$DB_PATH"
    --pessimism
    --skip_data_prep
    --run_tag "$RUN_TAG"
)
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Artifacts under:"
echo "  scratch: ${FWD_SCRATCH}/${MODEL}/${FRONT}/iteration_${ITER}/children_*_${RUN_TAG}/ehvi_values_*.csv"
echo "  batch:   ${FWD_SCRATCH}/${MODEL}/GENERATIONS/iteration_$((ITER + 1))/SIMULATIONS/simulation_candidates_gen$((ITER + 1))_${FRONT}_${RUN_TAG}.txt"

conda deactivate
