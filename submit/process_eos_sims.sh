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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/config/cluster.env"

module purge
module load "${CONDA_MODULE}"
module load "${INTEL_MODULE}" "${INTEL_MPI_MODULE}"
conda activate "${CONDA_ENV}"

MODEL=$1
NBOOT=$2
ITER=$3

if [[ $ITER -eq 0 ]]; then
    SEQS="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS/seq_init.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS"
    OUTPUT_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF"
else
    SEQS="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS/seq_gen$ITER.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS"
    OUTPUT_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF"
fi

python "${REPO_ROOT}/analysis/process_eos_sims.py" \
    -parent_dir "$PAR_DIR" \
    -output_dir "$PAR_DIR" \
    -sequence_file "$SEQS" \
    -num_bootstrap "$NBOOT"

cp "$PAR_DIR/eos_results.csv" "$OUTPUT_DIR/eos_results.csv"
cp "$SEQS" "$OUTPUT_DIR/seq_gen$ITER.txt"
