#!/bin/bash

#SBATCH --job-name=moe_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=test_suite.out
#SBATCH --error=test_suite.err


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


pytest tests/

LOG_DEST="${SLURM_SUBMIT_DIR:-.}/logs"

mkdir -p "$LOG_DEST"

SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/test_suite.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/test_suite.err"
[[ -f "$SLURM_OUT" ]] && mv "$SLURM_OUT" "$LOG_DEST/test_suite.out"
[[ -f "$SLURM_ERR" ]] && mv "$SLURM_ERR" "$LOG_DEST/test_suite.err"

conda deactivate

