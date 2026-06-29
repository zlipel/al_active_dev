#!/bin/bash
#SBATCH --job-name=process_diff
#SBATCH --output=simlogs/DIFF/process_diff.out
#SBATCH --error=simlogs/DIFF/process_diff.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:59
# Note: job-name/output/error are set by diff_calc.sh at submission time via sbatch CLI flags.
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
ITER=$2
INNER_JOBS=${3:-4}
OMP_THREADS=${4:-4}
NSEQ_JOBS=${5:-6}

if [[ $ITER -eq 0 ]]; then
    SEQS="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF/seq_init.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF"
else
    SEQS="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF/seq_gen$ITER.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF"
fi

OUTPUT_DIR="$PAR_DIR"

# Simulation analysis parameters
NRUNS=6
NSTEPS=15000000
DT=10.0
NCHAINS=100
NFREQ=1000
CUT=20
STRIDE=1

export OMP_NUM_THREADS=$OMP_THREADS

python "${REPO_ROOT}/analysis/process_diff_sims.py" \
    --parent_dir "$PAR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sequence_file "$SEQS" \
    --nruns $NRUNS \
    --nsteps $NSTEPS \
    --dt $DT \
    --nchains $NCHAINS \
    --nfreq $NFREQ \
    --inner_jobs $INNER_JOBS \
    --omp_threads $OMP_THREADS \
    --nseq_jobs $NSEQ_JOBS \
    --cutoff $CUT \
    --stride $STRIDE
