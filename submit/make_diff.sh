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
VALIDATION=""
CHECK_FIN=false
QUICK=""
CPUS_PER_SIM=""
MAX_CORES=""

usage() {
    cat <<EOF
Usage: $0 --model MODEL [SCOPE] [--nsim N] [--check_finished] [--quick N] [--cpus_per_sim N] [--max_cores N]

Scope (exactly one of):
  --init                            Initial-condition seed sims
                                    (reads \$SCRATCH_AL/<MODEL>/SIMULATIONS/DIFF/seq_init.txt)
  --iter N                          AL iteration N sims
                                    (reads \$SCRATCH_AL/<MODEL>/GENERATIONS/iteration_N/SIMULATIONS/DIFF/)
  --validation SCOPE                Post-hoc validation sims (e.g. --validation BENCHMARK)
                                    (reads \$SCRATCH_AL/<MODEL>/VALIDATION/SCOPE/SIMULATIONS/DIFF/seq_<scope-lower>.txt
                                     — populate first with gen_validation_sequences.py)

Optional:
  --nsim NSIM                       Independent production runs (default: 5)
  --check_finished                  Skip already-completed sims
  --quick N                         Fixed init density without EOS lookup
  --cpus_per_sim N                  CPUs per sim partition (default: make_diff.py = 16)
  --max_cores N                     Total core ceiling (default: make_diff.py = 480)
EOF
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --init) INIT=true ;;
        --iter) ITER="$2"; shift ;;
        --validation) VALIDATION="$2"; shift ;;
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
elif [[ -n "$VALIDATION" ]]; then
    SCOPE_LOWER="${VALIDATION,,}"
    PAR_DIR="${SCRATCH_AL}/$MODEL/VALIDATION/$VALIDATION/SIMULATIONS/DIFF"
    SEQS="$PAR_DIR/seq_${SCOPE_LOWER}.txt"
    LOG_TAG="validation_${SCOPE_LOWER}"
    if [[ ! -f "$SEQS" ]]; then
        echo "Error: sequence file not found at $SEQS" >&2
        echo "  Populate first via:" >&2
        echo "    python \"${REPO_ROOT}/beam_search/tools/gen_validation_sequences.py\" \\" >&2
        echo "        --scratch_dir \"\$SCRATCH_AL\" --length_changes --scope $VALIDATION" >&2
        exit 1
    fi
else
    if [[ -z "$ITER" ]]; then
        echo "Error: exactly one of --init / --iter / --validation is required"
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
