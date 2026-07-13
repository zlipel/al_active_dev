#!/bin/bash
#SBATCH --job-name=init_beams
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# MPI-parallel beam search over one model / mode / policy.
#
# Reads endpoint targets from
#   <SCRATCH_AL>/PATHS[_FIXED_LENGTH]/<MODEL>/<MODE>/endpoints_<MODEL>.csv
# and writes results (+ optional --profile timings) under
#   <SCRATCH_AL>/PATHS[_FIXED_LENGTH]/<MODEL>/<MODE>/<POLICY>/
#     ├── RESULTS/start_XXXX/paths.csv
#     └── step_timings/start_XXXX.csv        # only when --profile is set
#
# One rank is the conductor (start-index dispatcher); the rest are workers.
# Each worker holds an MoE bundle in memory and processes one start at a
# time. Set --ntasks such that (ntasks - 1) workers get enough starts each.
#
# For the benchmark (16 endpoints total per model at N_ps=N_nonps=2, 4
# diagonals): --ntasks=5 gives 4 workers and each processes 4 endpoints;
# 1 hour of wall time is plenty on CPU.
#
# For production (~64 targets × ~30-100 starts): --ntasks=8 workers × 3-4
# hours is a typical run.
#
# Usage (benchmark, expert_tied policy, --profile on):
#   sbatch --ntasks=5 --time=01:00:00 submit/run_beams.sh \
#       --model HPS_URRY --mode benchmark --policy expert_tied --profile
#
# Usage (production, expert_tied):
#   sbatch submit/run_beams.sh --model HPS_URRY --mode production \
#       --policy expert_tied

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
module load "${OPENMPI_MODULE}"
module load "${CONDA_MODULE}"
conda activate "${CONDA_ENV}"

# beam_search/ on PYTHONPATH so the runner's `from cross_paths.model_io ...`
# imports resolve; REPO_ROOT so `from al_pipeline.core.paths ...` resolves.
export PYTHONPATH="${REPO_ROOT}/beam_search:${REPO_ROOT}:${PYTHONPATH:-}"

# Numba: OMP threading layer inside each MPI rank; NUMBA_NUM_THREADS =
# SLURM_CPUS_PER_TASK so the featurizer uses every core allocated to this
# rank. The runner also calls nb.set_num_threads() at the top of each
# worker to align with SLURM_CPUS_PER_TASK — this env var is the fallback
# for the module-import path.
export NUMBA_THREADING_LAYER=omp
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Cap the numeric-library thread pools that torch / numpy would otherwise
# fan out into — MPI oversubscription otherwise. Torch's own thread count
# is set inside the runner (--torch_threads).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---- defaults ----
MODEL=""
MODE="benchmark"
POLICY="expert_tied"
FINAL_ITER=10
FRONT="upper"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
TRANSFORM="yeoj"

BEAM_WIDTH=32
MAX_STEPS=8
TOL_U="0.005"
TOL_V="0.005"
STAGNATION_PATIENCE=0
STAGNATION_DELTA="0.0"

FEAT_THREADS="${SLURM_CPUS_PER_TASK:-12}"
TORCH_THREADS="${SLURM_CPUS_PER_TASK:-12}"

HARD_THRESHOLD="0.5"
REJECT_THRESHOLD="0.5"

LENGTH_CHANGES=false
MC_EHVI=false
RESUME=false
EXTEND_NO_FINISHED=false
EXTRA_STEPS=0
PROFILE=false

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M [--mode {benchmark,production}]
                [--policy {expert_tied,anchored_reject,soft,hard,global}]
                [options]

Required:
  --model NAME              CALVADOS | HPS_URRY | MPIPI

Common options (defaults shown):
  --mode {benchmark,production}    (default: benchmark)
  --policy P                        (default: expert_tied)
  --final_iter N                    (default: 10)
  --front {upper,lower}             (default: upper)
  --beam_width N                    (default: 32)
  --max_steps N                     (default: 8)
  --tol_u F --tol_v F               finish tolerance in quantile space
                                    (default: 0.005 / 0.005)
  --stagnation_patience N           early-stop after N non-improving beam steps
                                    (default: 0 → disabled)
  --stagnation_delta F              minimum improvement to reset stagnation
                                    (default: 0.0)
  --feat_threads N                  numba threads per rank
                                    (default: SLURM_CPUS_PER_TASK)
  --torch_threads N                 torch threads per rank
                                    (default: SLURM_CPUS_PER_TASK)
  --hard_threshold F                gate threshold for --policy hard
                                    (default: 0.5)
  --reject_threshold F              gate threshold for --policy anchored_reject
                                    (default: 0.5)
  --length_changes                  Enable length-changing edits
  --mc_ehvi                         MC-EHVI checkpoint naming
  --resume                          Skip start_idx whose paths.csv is complete
  --extend_no_finished              Rerun no_finished endpoints with extra steps
  --extra_steps N                   Additional steps for no_finished retries
                                    (default: 0)
  --profile                         Log per-step featurize / predict_design ms
                                    to step_timings/start_XXXX.csv

Any other flag is forwarded verbatim to run_beams_mpi.py.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift ;;
        --mode)              MODE="$2"; shift ;;
        --policy)            POLICY="$2"; shift ;;
        --final_iter)        FINAL_ITER="$2"; shift ;;
        --front)             FRONT="$2"; shift ;;
        --ehvi_variant)      EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --transform)         TRANSFORM="$2"; shift ;;
        --beam_width)        BEAM_WIDTH="$2"; shift ;;
        --max_steps)         MAX_STEPS="$2"; shift ;;
        --tol_u)             TOL_U="$2"; shift ;;
        --tol_v)             TOL_V="$2"; shift ;;
        --stagnation_patience) STAGNATION_PATIENCE="$2"; shift ;;
        --stagnation_delta)  STAGNATION_DELTA="$2"; shift ;;
        --feat_threads)      FEAT_THREADS="$2"; shift ;;
        --torch_threads)     TORCH_THREADS="$2"; shift ;;
        --hard_threshold)    HARD_THRESHOLD="$2"; shift ;;
        --reject_threshold)  REJECT_THRESHOLD="$2"; shift ;;
        --length_changes)    LENGTH_CHANGES=true ;;
        --mc_ehvi)           MC_EHVI=true ;;
        --resume)            RESUME=true ;;
        --extend_no_finished) EXTEND_NO_FINISHED=true ;;
        --extra_steps)       EXTRA_STEPS="$2"; shift ;;
        --profile)           PROFILE=true ;;
        --help|-h)           usage ;;
        --)                  shift; EXTRA_FLAGS+=("$@"); break ;;
        *)                   EXTRA_FLAGS+=("$1") ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    usage
fi

CMD=(python -u "${REPO_ROOT}/beam_search/run_beams_mpi.py"
    --scratch_dir "$SCRATCH_AL"
    --home_dir    "$HOME_AL"
    --db_root     "$DB_ROOT"
    --model       "$MODEL"
    --mode        "$MODE"
    --policy      "$POLICY"
    --final_iter  "$FINAL_ITER"
    --front       "$FRONT"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --transform   "$TRANSFORM"
    --beam_width  "$BEAM_WIDTH"
    --max_steps   "$MAX_STEPS"
    --tol_u       "$TOL_U"
    --tol_v       "$TOL_V"
    --stagnation_patience "$STAGNATION_PATIENCE"
    --stagnation_delta    "$STAGNATION_DELTA"
    --feat_threads  "$FEAT_THREADS"
    --torch_threads "$TORCH_THREADS"
    --hard_threshold  "$HARD_THRESHOLD"
    --reject_threshold "$REJECT_THRESHOLD"
    --extra_steps "$EXTRA_STEPS"
)

[[ "$LENGTH_CHANGES"     == true ]] && CMD+=(--length_changes)
[[ "$MC_EHVI"            == true ]] && CMD+=(--mc_ehvi)
[[ "$RESUME"             == true ]] && CMD+=(--resume)
[[ "$EXTEND_NO_FINISHED" == true ]] && CMD+=(--extend_no_finished)
[[ "$PROFILE"            == true ]] && CMD+=(--profile)
CMD+=("${EXTRA_FLAGS[@]}")

NTASKS="${SLURM_NTASKS:-1}"
echo "Launching ${NTASKS} MPI ranks: ${CMD[*]}"
srun -n "$NTASKS" "${CMD[@]}"

# Route SLURM logs to the mode/policy subfolder. Header uses `%x_%j.out`
# so the scheduler-written filename is `<jobname>_<jobid>.out`; reconstruct
# that here to find and move it. Works whether the wrapper set --job-name
# to a per-model tag or we're using the header default.
LENGTH_DIR="PATHS_FIXED_LENGTH"
[[ "$LENGTH_CHANGES" == true ]] && LENGTH_DIR="PATHS"
LOG_DEST="${SCRATCH_AL}/${LENGTH_DIR}/${MODEL}/${MODE^^}/${POLICY}/logs"
mkdir -p "$LOG_DEST"
JOB_LOG_BASE="${SLURM_JOB_NAME:-init_beams}_${SLURM_JOB_ID:-local}"
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/${JOB_LOG_BASE}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/${JOB_LOG_BASE}.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/${JOB_LOG_BASE}.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/${JOB_LOG_BASE}.err"

conda deactivate
