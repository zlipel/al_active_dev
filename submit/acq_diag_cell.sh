#!/bin/bash
#SBATCH --job-name=acq_diag_cell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --output=acq_logs/acq_diag_%j.out
#SBATCH --error=acq_logs/acq_diag_%j.err
#
# One cell of the acquisition diagnostic: a single (model, iter, seed, method)
# run of the sequential batch acquisition against a frozen training snapshot, in
# a fully isolated tree on the acq side.
#
# Reads ONLY the staged snapshot produced by submit/stage_acq_snapshots.sh
# (acq side). It never opens a production path. All al-master output goes to the
# isolated cell tree under HOME_AL_ACQ / SCRATCH_AL_ACQ.
#
# Method maps to the condition under test:
#   kb           kriging-believer, believed label = posterior mean (no pessimism)
#   pkb          kriging-believer + pessimism (believed label shifted by overlap)
#   independent  --exploration_strategy standard: no within-batch update; every
#                pick scores against the unchanged base surrogate
#
# Reuses the production `al-master` entry point (same code path as a real AL
# round) with --skip_data_prep + --acq_test, as submit/forward_round.sh does.
#
# Usage (normally launched by submit/run_acq_diag.sh):
#   sbatch submit/acq_diag_cell.sh --model MPIPI --iter 6 --seed 12345 --method pkb \
#       --exp_name seminar_20260923

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/config/cluster.env" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${AL_ACTIVE_DEV:-}" && -f "${AL_ACTIVE_DEV}/config/cluster.env" ]]; then
    REPO_ROOT="${AL_ACTIVE_DEV}"
else
    REPO_ROOT="${HOME}/PROJECTS/al_active_dev"
fi
source "${REPO_ROOT}/config/cluster.env"
source "${REPO_ROOT}/submit/acq_diag_lib.sh"

acq_activate_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL=""; ITER=""; SEED=""; METHOD=""
FRONT="upper"
EXP_NAME="seminar_$(date +%Y%m%d)"
PILOT=false

# Held fixed across every cell so only (iter, method, seed) vary.
TRAIN_MODEL_TYPE="gpr_multitask"
TRANSFORM="yeoj"
EHVI_VARIANT="epsilon"
EPSILON_SCALE="2.0"
REF_POINT_MODE="frac"
REF_POINT_FRAC="0.5"
REF_POINT_TAU="0.05"
REF_POINT_CAP="0.5"
OBJ1="exp_density"
OBJ2="diff"

usage() {
    cat <<EOF
Usage: sbatch $0 --model M --iter N --seed S --method {kb,pkb,independent} [options]

Required:
  --model NAME               Force field (e.g. MPIPI)
  --iter  N                  Frozen snapshot / round (proposes round N+1)
  --seed  S                  GA seed_base (e.g. 12345)
  --method {kb,pkb,independent}

Options:
  --front {upper,lower}      (default: upper)
  --exp_name NAME            Experiment group (default: seminar_YYYYMMDD)
  --pilot                    Tiny budget (ngen 4, ncands 8, ga_max_iter 20)

The staged snapshot for (model, iter, exp_name) must already exist
(submit/stage_acq_snapshots.sh).
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2"; shift ;;
        --iter)     ITER="$2"; shift ;;
        --seed)     SEED="$2"; shift ;;
        --method)   METHOD="$2"; shift ;;
        --front)    FRONT="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --pilot)    PILOT=true ;;
        --help|-h)  usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

[[ -z "$MODEL"  ]] && { echo "Error: --model is required";  usage; }
[[ -z "$ITER"   ]] && { echo "Error: --iter is required";   usage; }
[[ -z "$SEED"   ]] && { echo "Error: --seed is required";   usage; }
[[ -z "$METHOD" ]] && { echo "Error: --method is required"; usage; }
acq_validate_name "$MODEL" model
acq_validate_name "$EXP_NAME" exp_name
acq_validate_integer "$ITER" iteration
acq_validate_integer "$SEED" seed
[[ "$FRONT" == upper || "$FRONT" == lower ]] || { echo "Invalid front: $FRONT"; exit 2; }

# Hard isolation guard: acq roots must be separate from production roots.
assert_acq_roots_isolated

# ── Budget ──────────────────────────────────────────────────────────────────
if [[ "$PILOT" == true ]]; then
    NGEN=4;  NCANDS=8;  GA_MAX_ITER=20;  GA_MAX_NO_IMPROV=10
else
    NGEN=24; NCANDS=96; GA_MAX_ITER=200; GA_MAX_NO_IMPROV=50
fi

# ── Method -> flags ─────────────────────────────────────────────────────────
METHOD_FLAGS=()
case "$METHOD" in
    kb)          METHOD_FLAGS=(--exploration_strategy kriging_believer) ;;
    pkb)         METHOD_FLAGS=(--exploration_strategy kriging_believer --pessimism) ;;
    independent) METHOD_FLAGS=(--exploration_strategy standard) ;;
    *) echo "Error: --method must be kb, pkb or independent"; usage ;;
esac

# ── Cell isolation (acq side only) ──────────────────────────────────────────
CELL=$(acq_cell_name "$MODEL" "$ITER" "$SEED" "$METHOD" "$FRONT" "$PILOT")
CELL_HOME="${HOME_AL_ACQ}/${EXP_NAME}/${CELL}"
CELL_SCRATCH="${SCRATCH_AL_ACQ}/${EXP_NAME}/${CELL}"

# Belt-and-suspenders: both cell roots must resolve on the acq side.
assert_under_acq_roots "$CELL_HOME" "cell home"
assert_under_acq_roots "$CELL_SCRATCH" "cell scratch"

# Never clobber an existing cell — re-runs use a new exp_name or delete first.
if [[ -e "$CELL_HOME" || -e "$CELL_SCRATCH" ]]; then
    echo "Refusing to overwrite existing diagnostic cell: ${CELL}"
    echo "  home:    ${CELL_HOME}"
    echo "  scratch: ${CELL_SCRATCH}"
    exit 2
fi

# Seed source is the READ-ONLY staged snapshot (acq side), not production.
STAGE_DIR=$(acq_snapshot_dir "$EXP_NAME" "$MODEL" "$ITER")
if [[ ! -d "$STAGE_DIR" ]]; then
    echo "Error: no staged snapshot at ${STAGE_DIR}"
    echo "  run: ./submit/stage_acq_snapshots.sh --model ${MODEL} --iters \"${ITER}\" --exp_name ${EXP_NAME}"
    exit 1
fi
assert_under_acq_roots "$STAGE_DIR" "snapshot"
python "${REPO_ROOT}/submit/acq_diag_tools.py" verify --directory "$STAGE_DIR" --iteration "$ITER"

DEST_DIR="${CELL_SCRATCH}/${MODEL}/GENERATIONS/iteration_${ITER}"
mkdir -p "$(dirname "$CELL_HOME")"
mkdir "$CELL_HOME"   # atomic cell claim: duplicate jobs cannot both proceed
mkdir -p "$DEST_DIR"

echo "Seeding cell from staged snapshot: ${STAGE_DIR} -> ${DEST_DIR}"
for name in $(acq_snapshot_files "$ITER"); do
    if [[ ! -f "${STAGE_DIR}/${name}" ]]; then
        echo "Error: staged snapshot missing ${STAGE_DIR}/${name}"
        exit 1
    fi
    cp "${STAGE_DIR}/${name}" "${DEST_DIR}/${name}"
    chmod u+w "${DEST_DIR}/${name}"   # staged copy is read-only; cell copy is writable
done
cp "${STAGE_DIR}/snapshot_manifest.json" "${DEST_DIR}/snapshot_manifest.json"
python "${REPO_ROOT}/submit/acq_diag_tools.py" verify --directory "$DEST_DIR" --iteration "$ITER"
cp "${STAGE_DIR}/snapshot_manifest.json" "${CELL_HOME}/snapshot_manifest.json"

# ── Provenance record ───────────────────────────────────────────────────────
git -C "$REPO_ROOT" rev-parse HEAD > "${CELL_HOME}/code_commit.txt" 2>/dev/null || \
    echo "unknown" > "${CELL_HOME}/code_commit.txt"

# Master training initialization is fixed across methods and GA seeds.
CMD=(python "${REPO_ROOT}/submit/acq_diag_tools.py" seeded-master --master-seed 271828 --
    --model "$MODEL"
    --iter "$ITER"
    --front "$FRONT"
    --train_model_type "$TRAIN_MODEL_TYPE"
    --transform "$TRANSFORM"
    --ehvi_variant "$EHVI_VARIANT"
    --epsilon_scale "$EPSILON_SCALE"
    --ref_point_mode "$REF_POINT_MODE"
    --ref_point_frac "$REF_POINT_FRAC"
    --ref_point_tau "$REF_POINT_TAU"
    --ref_point_cap "$REF_POINT_CAP"
    --obj1 "$OBJ1"
    --obj2 "$OBJ2"
    --seed_base "$SEED"
    --ngen "$NGEN"
    --ncands "$NCANDS"
    --ga_max_iter "$GA_MAX_ITER"
    --ga_max_no_improv "$GA_MAX_NO_IMPROV"
    --base_path "$CELL_HOME"
    --scratch_path "$CELL_SCRATCH"
    --db_path "$DB_PATH"
    --skip_data_prep
    --acq_test
    "${METHOD_FLAGS[@]}"
)
python - "$REPO_ROOT" "$CELL_HOME" <<'PY'
from pathlib import Path
import hashlib, json, sys
repo, output = map(Path, sys.argv[1:])
files = list((repo / "al_pipeline").rglob("*.py")) + list((repo / "submit").glob("*acq*"))
manifest = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}
(output / "code_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

{ printf '%q ' "${CMD[@]}"; printf '\n'; } > "${CELL_HOME}/command.txt"

echo "Cell: ${CELL}  (pilot=${PILOT})"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Cell artifacts:"
echo "  ehvi curve: ${CELL_SCRATCH}/${MODEL}/${FRONT}/iteration_${ITER}/children_*/ehvi_values_*.csv"
echo "  diversity:  ${CELL_HOME}/${MODEL}/DIAGNOSTIC/batch_diversity_*.csv"
echo "  batch:      ${CELL_SCRATCH}/${MODEL}/GENERATIONS/iteration_$((ITER + 1))/SIMULATIONS/simulation_candidates_gen$((ITER + 1))_${FRONT}.txt"

conda deactivate
