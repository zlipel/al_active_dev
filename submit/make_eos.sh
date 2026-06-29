#!/bin/bash
#SBATCH --job-name=make_eos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=200MB
#SBATCH --time=02:59:59
#SBATCH --mail-type=${SBATCH_MAIL_TYPE}
#SBATCH --mail-user=${SBATCH_MAIL_USER}
#SBATCH --output=make_eos.out
#SBATCH --error=make_eos.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/config/cluster.env"

module purge
module load "${INTEL_MODULE}" "${INTEL_MPI_MODULE}"
module load "${CONDA_MODULE}"
conda activate "${CONDA_ENV}"

MODEL=""
ITER=""
RHO_i=""
RHO_f=""
DRHO=""
INIT=false
CHECK_RHO=false
CHECK_FINISHED=false
CPUS_PER_SIM=""
MAX_CORES=""

usage() {
    echo "Usage: $0 [--init] --model MODEL --iter ITER --rho_i RHO_i --rho_f RHO_f --drho DRHO [--check_rho] [--check_finished] [--cpus_per_sim N] [--max_cores N]"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --init) INIT=true ;;
        --iter) ITER="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --rho_i) RHO_i="$2"; shift ;;
        --rho_f) RHO_f="$2"; shift ;;
        --drho) DRHO="$2"; shift ;;
        --check_rho) CHECK_RHO=true ;;
        --check_finished) CHECK_FINISHED=true ;;
        --cpus_per_sim) CPUS_PER_SIM="$2"; shift ;;
        --max_cores) MAX_CORES="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL" || -z "$RHO_i" || -z "$RHO_f" || -z "$DRHO" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

if [[ "$INIT" == true ]]; then
    SEQS="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS/seq_init.txt"
    PAR_DIR="${SCRATCH_AL}/$MODEL/SIMULATIONS/EOS/"
    LOG_TAG="init"
else
    if [[ -z "$ITER" ]]; then
        echo "Error: --iter is required when --init is not set"
        usage
    fi
    PAR_DIR="${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/SIMULATIONS/EOS"

    # al_pipeline mirrors candidates into EOS/ at these paths (primary location):
    SEQ_UPPER="$PAR_DIR/simulation_candidates_gen${ITER}_upper.txt"
    SEQ_LOWER="$PAR_DIR/simulation_candidates_gen${ITER}_lower.txt"
    SEQS="$PAR_DIR/seq_gen${ITER}.txt"

    # Combine whichever fronts exist; error if neither found.
    if [[ -f "$SEQ_UPPER" && -f "$SEQ_LOWER" ]]; then
        cat "$SEQ_UPPER" "$SEQ_LOWER" > "$SEQS"
    elif [[ -f "$SEQ_UPPER" ]]; then
        echo "Note: only upper-front candidates found for iteration $ITER; lower-front absent."
        cp "$SEQ_UPPER" "$SEQS"
    elif [[ -f "$SEQ_LOWER" ]]; then
        echo "Note: only lower-front candidates found for iteration $ITER; upper-front absent."
        cp "$SEQ_LOWER" "$SEQS"
    else
        echo "Error: no candidate sequence files found for iteration $ITER in $PAR_DIR" >&2
        exit 1
    fi

    cp "$SEQS" "${SCRATCH_AL}/$MODEL/GENERATIONS/iteration_$ITER/seq_gen${ITER}.txt"
    LOG_TAG="gen$ITER"
fi

LOGS="$PAR_DIR/logs/"
mkdir -p "$LOGS"
cd "$PAR_DIR"   # polymerize / gendata drop temp files into cwd

CMD="python \"${REPO_ROOT}/simulation/make_eos.py\" --model \"$MODEL\" --num_polymers 100 \
    --density_start \"$RHO_i\" --density_end \"$RHO_f\" --density_step \"$DRHO\" \
    --sequence_file \"$SEQS\" --parent_dir \"$PAR_DIR\""
[[ "$CHECK_RHO" == true ]] && CMD+=" --check_densities"
[[ "$CHECK_FINISHED" == true ]] && CMD+=" --check_finished"
[[ -n "$CPUS_PER_SIM" ]] && CMD+=" --cpus_per_sim $CPUS_PER_SIM"
[[ -n "$MAX_CORES" ]] && CMD+=" --max_cores $MAX_CORES"
eval "$CMD"

mv "make_eos.out" "$LOGS/make_eos_${LOG_TAG}.out"
mv "make_eos.err" "$LOGS/make_eos_${LOG_TAG}.err"

conda deactivate
