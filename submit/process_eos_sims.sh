#!/bin/bash
#SBATCH --job-name=process_eos
#SBATCH --output=simlogs/EOS/process_eos.out
#SBATCH --error=simlogs/EOS/process_eos.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=00:29:59
# Note: job-name/output/error are set by eos_calc.sh at submission time via sbatch CLI flags.
# The #SBATCH values above are generic defaults and will be overridden.

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
module load "${INTEL_MODULE}" "${INTEL_MPI_MODULE}"
conda activate "${CONDA_ENV}"

MODEL=$1
NBOOT=$2
# Third arg is the scope selector: an integer iteration (0 = init) or the
# token "validation:SCOPE" emitted by eos_calc.sh --validation SCOPE. The
# validation case must be tested first — [[ ... -eq 0 ]] is arithmetic and
# would (mis)evaluate a "validation:*" string to 0.
SCOPE_SPEC=$3
# Fourth arg is the optional parallel-policy run tag (iteration scope only); the
# AL loop appends _<run_tag> to the seq filename so parallel policies sharing one
# iteration dir don't collide. Empty ⇒ production naming (backward compatible).
RUN_TAG="${4:-}"
TAG_SUFFIX=""
[[ -n "$RUN_TAG" ]] && TAG_SUFFIX="_${RUN_TAG}"

if [[ "$SCOPE_SPEC" == validation:* ]]; then
    SCOPE="${SCOPE_SPEC#validation:}"
    SCOPE_LOWER="${SCOPE,,}"
    PAR_DIR="${SCRATCH_AL}/$MODEL/VALIDATION/$SCOPE/SIMULATIONS/EOS"
    OUTPUT_DIR="${SCRATCH_AL}/$MODEL/VALIDATION/$SCOPE/SIMULATIONS/DIFF"
    SEQS="$PAR_DIR/seq_${SCOPE_LOWER}.txt"
elif [[ $SCOPE_SPEC -eq 0 ]]; then
    SEQS="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS/seq_init.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS"
    OUTPUT_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF"
else
    ITER="$SCOPE_SPEC"
    SEQS="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS/seq_gen${ITER}${TAG_SUFFIX}.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS"
    OUTPUT_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF"
fi

python "${REPO_ROOT}/analysis/process_eos_sims.py" \
    -parent_dir "$PAR_DIR" \
    -output_dir "$PAR_DIR" \
    -sequence_file "$SEQS" \
    -num_bootstrap "$NBOOT"

# make_diff.py reads eos_results.csv from its parent_dir (the DIFF dir), so
# hand the EOS results across to the sibling DIFF tree.
cp "$PAR_DIR/eos_results.csv" "$OUTPUT_DIR/eos_results.csv"
if [[ "$SCOPE_SPEC" == validation:* ]]; then
    cp "$SEQS" "$OUTPUT_DIR/seq_${SCOPE_LOWER}.txt"
else
    cp "$SEQS" "$OUTPUT_DIR/seq_gen${SCOPE_SPEC}${TAG_SUFFIX}.txt"
fi
