#!/bin/bash
#SBATCH --job-name=resume_beams
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:59
#SBATCH --mem-per-cpu=4G
#SBATCH --output=resume_beams.out
#SBATCH --error=resume_beams.err
# Note: job-name/output/error are set by submit_resume_beams.sh at submission time.

source "${HOME}/PROJECTS/al_active_dev/config/cluster.env"
module purge
module load "${CONDA_MODULE}"
module load "${OPENMPI_MODULE}"
conda activate "${CONDA_ENV}"

export NUMBA_THREADING_LAYER=omp
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FEAT_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=$1
ITER=$2
LENGTH_CHANGES=$3
EXTEND_NO_FINISHED=${4:-false}
EXTRA_STEPS=${5:-0}
STAGNATION_PATIENCE=${6:-0}
STAGNATION_DELTA=${7:-0.0}

FRONT=${8:-upper}
EHVI_VARIANT=${9:-epsilon}
EXPLORATION_STRATEGY=${10:-kriging_believer}
TRANSFORM=${11:-yeoj}
MC_EHVI=${12:-false}

if [[ "$LENGTH_CHANGES" == "true" ]]; then
  PATHS_DIR="${SCRATCH_AL}/PATHS/$MODEL"
else
  PATHS_DIR="${SCRATCH_AL}/PATHS_FIXED_LENGTH/$MODEL"
fi

echo "Resume mode: using existing results in ${PATHS_DIR}"
echo "No files will be deleted and endpoints will not be regenerated."

CMD="srun --cpu-bind=cores --distribution=block:block python run_beams_mpi.py \
  --scratch_dir          $SCRATCH_AL \
  --home_dir             $HOME_AL \
  --db_root              $DB_ROOT \
  --model                $MODEL \
  --final_iter           $ITER \
  --front                $FRONT \
  --ehvi_variant         $EHVI_VARIANT \
  --exploration_strategy $EXPLORATION_STRATEGY \
  --transform            $TRANSFORM \
  --feat_threads         $FEAT_THREADS \
  --torch_threads        $OMP_NUM_THREADS \
  --beam_width           32 \
  --max_steps            80 \
  --tol_u                0.002 \
  --tol_v                0.002 \
  --resume"

[[ "$MC_EHVI" == "true" ]]             && CMD+=" --mc_ehvi"
[[ "$LENGTH_CHANGES" == "true" ]]      && CMD+=" --length_changes"
[[ "$EXTEND_NO_FINISHED" == "true" ]]  && CMD+=" --extend_no_finished"
[[ "$EXTRA_STEPS" != "0" ]]            && CMD+=" --extra_steps $EXTRA_STEPS"
[[ "$STAGNATION_PATIENCE" != "0" ]]    && CMD+=" --stagnation_patience $STAGNATION_PATIENCE"
[[ "$STAGNATION_DELTA" != "0.0" ]]     && CMD+=" --stagnation_delta $STAGNATION_DELTA"

echo "$CMD"
eval $CMD

echo "Resume beam searches complete, collecting results..."

CMD="python collect_results.py \
  --scratch_dir $SCRATCH_AL \
  --model       $MODEL"
[[ "$LENGTH_CHANGES" == "true" ]] && CMD+=" --length_changes"
eval $CMD
