#!/bin/bash
#SBATCH --job-name=eos_relabel
#SBATCH --output=eos_relabel_%j.out
#SBATCH --error=eos_relabel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:59:59
# Pass-through wrapper for analysis/eos_relabel.py. Everything after the script
# name is forwarded verbatim, so the report/apply steps are separate sbatch jobs:
#
#   sbatch submit/eos_relabel.sh report
#   #   review analysis/_anomaly_step3/eos_diff_report.csv, set approved=1 on kept rows
#   sbatch submit/eos_relabel.sh apply --from-report analysis/_anomaly_step3/eos_diff_report.csv          # dry-run
#   sbatch submit/eos_relabel.sh apply --from-report analysis/_anomaly_step3/eos_diff_report.csv --apply  # write
#
# The report scans thermo.avg + diff log.lammps for every available polymer and
# parallelises across the allocated cores (joblib). apply is light but is routed
# through SLURM too so nothing runs on the login node. Logs land as
# eos_relabel_<jobid>.{out,err} in the submit directory (gitignored).

set -eo pipefail

# Resolve repo root. SLURM copies the script to a spool dir, so BASH_SOURCE/$0
# don't self-locate; prefer SLURM_SUBMIT_DIR, then AL_ACTIVE_DEV, then canonical.
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

# numpy 2.x C-extensions need GLIBCXX_3.4.29 from the conda libstdc++, not
# /lib64's older one. Same fixup the beam/prep submit scripts apply.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# joblib spawns one process per core; keep each single-threaded so numpy/pandas
# threads don't oversubscribe.
export OMP_NUM_THREADS=1

# cd so repo-relative paths (e.g. --from-report analysis/...) resolve, and the
# report's default --out lands under the repo tree.
cd "${REPO_ROOT}"

python "${REPO_ROOT}/analysis/eos_relabel.py" "$@"

conda deactivate
