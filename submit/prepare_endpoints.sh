#!/bin/bash
#SBATCH --job-name=prep_endpoints
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Prepare beam-search endpoint targets for one model.
#
# Reads the iter-N labels + the final full-data MoE bundle, builds threshold
# pools (regime + p_ps), stratifies each pool by (u, v) property quantiles,
# picks starts per mode, generates the target-delta grid, and writes:
#
#   <SCRATCH_AL>/PATHS[_FIXED_LENGTH]/<MODEL>/<MODE>/
#     ├── endpoints_<MODEL>.csv
#     ├── starts_<MODEL>.csv
#     └── config.json
#
# The runner (run_beams.sh) consumes endpoints_<MODEL>.csv from that same
# folder via matching --mode.
#
# Usage:
#   sbatch submit/prepare_endpoints.sh --model HPS_URRY --mode benchmark
#   sbatch submit/prepare_endpoints.sh --model MPIPI --mode production \
#       --frac_ps 0.9 --frac_nonps 0.75

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

# Conda ships a newer libstdc++ than Stellar's /lib64/libstdc++.so.6 —
# numpy 2.x needs GLIBCXX_3.4.29 which the system lib lacks. Prepend
# conda's libdir so numpy's C extensions load. Must happen AFTER
# `conda activate` so $CONDA_PREFIX is set.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# prepare_endpoints.py uses `from cross_paths.model_io import ...` (unqualified),
# so `beam_search/` needs to be on PYTHONPATH. REPO_ROOT itself is also on
# PYTHONPATH so `from al_pipeline.core.paths import ALPaths` resolves.
export PYTHONPATH="${REPO_ROOT}/beam_search:${REPO_ROOT}:${PYTHONPATH:-}"

# Cap threaded numeric libraries for the featurizer sanity check +
# QT / RF calls — all single-batched here, no need to fan out.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---- defaults ----
MODEL=""
MODE="benchmark"
FINAL_ITER=10
FRONT="upper"
EHVI_VARIANT="epsilon"
EXPLORATION_STRATEGY="kriging_believer"
TRANSFORM="yeoj"

THRESH_LOWER="0.25"
THRESH_HIGHER="0.75"

PS_BINS=3
NONPS_BINS=5

GRID_SPACING="0.0125"
LARGEST_DELTA="0.05"
BENCHMARK_DELTA="0.0375"

N_PS=2
N_NONPS=2
FRAC_PS="0.9"
FRAC_NONPS="0.75"

SEED=0
LENGTH_CHANGES=false
MC_EHVI=false
CLEAR=false
EXCLUDE_STARTS=""

EXTRA_FLAGS=()

usage() {
    cat <<EOF
Usage: sbatch $0 --model M [--mode {benchmark,production}] [options]

Required:
  --model NAME              CALVADOS | HPS_URRY | MPIPI

Common options (defaults shown):
  --mode {benchmark,production}    (default: benchmark)
  --final_iter N                    (default: 10)
  --front {upper,lower}             (default: upper)
  --thresh_lower F                  nonPS pool cap (default: 0.25)
  --thresh_higher F                 PS pool floor (default: 0.75)
  --ps_bins N                       (default: 3)
  --nonps_bins N                    (default: 5)
  --grid_spacing F                  (default: 0.0125)
  --largest_delta F                 (default: 0.05)
  --benchmark_delta F               (default: 0.0375)
  --n_ps N / --n_nonps N            benchmark counts (default: 2 / 2)
  --frac_ps F / --frac_nonps F      production fractions (default: 0.9 / 0.75)
  --seed N                          RNG seed (default: 0)
  --length_changes                  Enable length-changing edits
  --mc_ehvi                         MC-EHVI checkpoint naming
  --clear                           Wipe the target <MODE> folder before writing
                                    (removes stale RESULTS/step_timings from a
                                    previous prep+beam cycle).
  --exclude_starts LIST             Comma-separated seq indices to drop from
                                    both pools before stratification. Use to
                                    replace a stuck start (all targets
                                    no_finished) with a different member of
                                    the same (u, v) bin.

Any other flag is forwarded verbatim to prepare_endpoints.py.
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)             MODEL="$2"; shift ;;
        --mode)              MODE="$2"; shift ;;
        --final_iter)        FINAL_ITER="$2"; shift ;;
        --front)             FRONT="$2"; shift ;;
        --ehvi_variant)      EHVI_VARIANT="$2"; shift ;;
        --exploration_strategy) EXPLORATION_STRATEGY="$2"; shift ;;
        --transform)         TRANSFORM="$2"; shift ;;
        --thresh_lower)      THRESH_LOWER="$2"; shift ;;
        --thresh_higher)     THRESH_HIGHER="$2"; shift ;;
        --ps_bins)           PS_BINS="$2"; shift ;;
        --nonps_bins)        NONPS_BINS="$2"; shift ;;
        --grid_spacing)      GRID_SPACING="$2"; shift ;;
        --largest_delta)     LARGEST_DELTA="$2"; shift ;;
        --benchmark_delta)   BENCHMARK_DELTA="$2"; shift ;;
        --n_ps)              N_PS="$2"; shift ;;
        --n_nonps)           N_NONPS="$2"; shift ;;
        --frac_ps)           FRAC_PS="$2"; shift ;;
        --frac_nonps)        FRAC_NONPS="$2"; shift ;;
        --seed)              SEED="$2"; shift ;;
        --length_changes)    LENGTH_CHANGES=true ;;
        --mc_ehvi)           MC_EHVI=true ;;
        --clear)             CLEAR=true ;;
        --exclude_starts)    EXCLUDE_STARTS="$2"; shift ;;
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

CMD=(python "${REPO_ROOT}/beam_search/prepare_endpoints.py"
    --scratch_dir "$SCRATCH_AL"
    --home_dir    "$HOME_AL"
    --db_root     "$DB_ROOT"
    --model       "$MODEL"
    --final_iter  "$FINAL_ITER"
    --front       "$FRONT"
    --ehvi_variant "$EHVI_VARIANT"
    --exploration_strategy "$EXPLORATION_STRATEGY"
    --transform   "$TRANSFORM"
    --mode        "$MODE"
    --thresh_lower  "$THRESH_LOWER"
    --thresh_higher "$THRESH_HIGHER"
    --ps_bins       "$PS_BINS"
    --nonps_bins    "$NONPS_BINS"
    --grid_spacing  "$GRID_SPACING"
    --largest_delta "$LARGEST_DELTA"
    --benchmark_delta "$BENCHMARK_DELTA"
    --seed        "$SEED"
)

# Mode-specific params.
if [[ "$MODE" == "benchmark" ]]; then
    CMD+=(--n_ps "$N_PS" --n_nonps "$N_NONPS")
else
    CMD+=(--frac_ps "$FRAC_PS" --frac_nonps "$FRAC_NONPS")
fi

[[ "$LENGTH_CHANGES" == true ]] && CMD+=(--length_changes)
[[ "$MC_EHVI"        == true ]] && CMD+=(--mc_ehvi)
[[ "$CLEAR"          == true ]] && CMD+=(--clear)
[[ -n "$EXCLUDE_STARTS" ]]      && CMD+=(--exclude_starts "$EXCLUDE_STARTS")
CMD+=("${EXTRA_FLAGS[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# Route SLURM logs to the mode subfolder for easy tracing. Header uses
# `%x_%j.out` so the scheduler-written filename is `<jobname>_<jobid>.out`
# — reconstruct that here to find and move it.
LENGTH_DIR="PATHS_FIXED_LENGTH"
[[ "$LENGTH_CHANGES" == true ]] && LENGTH_DIR="PATHS"
LOG_DEST="${SCRATCH_AL}/${LENGTH_DIR}/${MODEL}/${MODE^^}/logs"
mkdir -p "$LOG_DEST"
JOB_LOG_BASE="${SLURM_JOB_NAME:-prepare_endpoints}_${SLURM_JOB_ID:-local}"
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/${JOB_LOG_BASE}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/${JOB_LOG_BASE}.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/${JOB_LOG_BASE}.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/${JOB_LOG_BASE}.err"

conda deactivate
