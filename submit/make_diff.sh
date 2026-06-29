#!/bin/bash
#SBATCH --job-name=make_diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=100MB
#SBATCH --time=00:29:59
#SBATCH --output=make_diff.out
#SBATCH --error=make_diff.err
# To get failure emails, export SBATCH_MAIL_USER and SBATCH_MAIL_TYPE in your
# shell rc; sbatch picks them up from the environment automatically. Or pass
# --mail-user/--mail-type on the sbatch CLI at submit time. (#SBATCH headers
# are parsed by SLURM before the script runs and do NOT expand bash variables.)

set -eo pipefail

# Resolve repo root. SLURM copies the script to /var/spool/slurmd/... so
# BASH_SOURCE/$0 don't self-locate here. Prefer SLURM_SUBMIT_DIR (cwd from
# which sbatch was invoked), fall back to an exported AL_ACTIVE_DEV, then to
# the canonical install path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/config/cluster.env" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -n "${AL_ACTIVE_DEV:-}" && -f "${AL_ACTIVE_DEV}/config/cluster.env" ]]; then
    REPO_ROOT="${AL_ACTIVE_DEV}"
else
    REPO_ROOT="${HOME}/PROJECTS/al_active_dev"
fi
source "${REPO_ROOT}/config/cluster.env"

module purge
module load "${INTEL_MODULE}" "${INTEL_MPI_MODULE}"
module load "${CONDA_MODULE}"
conda activate "${CONDA_ENV}"

MODEL=""
ITER=""
NSIM="5"
INIT=false
CHECK_FIN=false
QUICK=""
CPUS_PER_SIM=""
MAX_CORES=""

usage() {
    echo "Usage: $0 --model MODEL [--iter ITER] [--init] [--nsim NSIM] [--check_finished] [--quick N] [--cpus_per_sim N] [--max_cores N]"
    echo "  --model MODEL        : Name of model used (e.g., hps_urry, mpipi)"
    echo "  --iter ITER          : Active learning iteration (if not init)"
    echo "  --init               : Whether these are initial sequences"
    echo "  --nsim NSIM          : Number of independent production simulations (default: 5)"
    echo "  --check_finished     : Check if simulations are all done"
    echo "  --quick N            : Quick mode (use fixed init density without EOS lookup)"
    echo "  --cpus_per_sim N     : CPUs per simulation partition (default: 16)"
    echo "  --max_cores N        : Total core ceiling for this submission (default: 480)"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --init) INIT=true ;;
        --iter) ITER="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --nsim) NSIM="$2"; shift ;;
        --check_finished) CHECK_FIN=true ;;
        --quick) QUICK="$2"; shift ;;
        --cpus_per_sim) CPUS_PER_SIM="$2"; shift ;;
        --max_cores) MAX_CORES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model must be specified"
    usage
fi

if [[ "$INIT" == true ]]; then
    SEQS="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF/seq_init.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/DIFF/"
    LOG_TAG="init"
else
    if [[ -z "$ITER" ]]; then
        echo "Error: --iter is required when --init is not set"
        usage
    fi
    cp "${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS/seq_gen$ITER.txt" \
       "${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF/seq_gen$ITER.txt"

    SEQS="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF/seq_gen$ITER.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/DIFF/"
    LOG_TAG="gen$ITER"
fi

LOGS="$PAR_DIR/logs/"
mkdir -p "$LOGS"
cd "$PAR_DIR"   # polymerize / gendata drop temp files into cwd

CMD="python \"${REPO_ROOT}/simulation/make_diff.py\" --model \"$MODEL\" \
    --num_polymers 100 \
    --nsim \"$NSIM\" \
    --sequence_file \"$SEQS\" \
    --parent_dir \"$PAR_DIR\" \
    --quick \"${QUICK:-0}\" "

[[ "$CHECK_FIN" == true ]] && CMD+=" --check_finished"
[[ -n "$CPUS_PER_SIM" ]] && CMD+=" --cpus_per_sim $CPUS_PER_SIM"
[[ -n "$MAX_CORES" ]] && CMD+=" --max_cores $MAX_CORES"
eval "$CMD"

# SLURM keeps --output / --error open in SLURM_SUBMIT_DIR, even after we cd.
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/make_diff.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/make_diff.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOGS/make_diff_${LOG_TAG}.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOGS/make_diff_${LOG_TAG}.err"

conda deactivate
