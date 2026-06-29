#!/bin/bash
#SBATCH --job-name=run_beams
#SBATCH --nodes=2
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:59
#SBATCH --mem-per-cpu=4G
#SBATCH --output=run_beams.out
#SBATCH --error=run_beams.err
# Note: job-name/output/error are set by submit_beams.sh at submission time.

source "${HOME}/PROJECTS/al_active_dev/config/cluster.env"
module purge
module load "${CONDA_MODULE}"
module load "${OPENMPI_MODULE}"
conda activate "${CONDA_ENV}"

# Thread settings per rank
export NUMBA_THREADING_LAYER=omp
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FEAT_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL=$1
ITER=$2
NBINS=$3
KPERBIN=$4
LENGTH_CHANGES=$5
EXTEND_NO_FINISHED=${6:-false}
EXTRA_STEPS=${7:-0}
STAGNATION_PATIENCE=${8:-0}
STAGNATION_DELTA=${9:-0.0}

# ALPaths args for checkpoint resolution
FRONT=${10:-upper}
EHVI_VARIANT=${11:-epsilon}
EXPLORATION_STRATEGY=${12:-kriging_believer}
TRANSFORM=${13:-yeoj}
MC_EHVI=${14:-false}

if [[ "$LENGTH_CHANGES" == "true" ]]; then
  PATHS_DIR="${SCRATCH_AL}/PATHS/$MODEL"
else
  PATHS_DIR="${SCRATCH_AL}/PATHS_FIXED_LENGTH/$MODEL"
fi

# NOTE: rm -rf ${PATHS_DIR}/* has been REMOVED.
# To clear existing results, pass --clear_paths to submit_beams.sh
# or manually delete $PATHS_DIR before submission.

# 1) Prepare endpoints (cheap, single-node)
python prepare_endpoints.py \
  --model       "$MODEL" \
  --scratch_dir "$SCRATCH_AL" \
  --home_dir    "$HOME_AL" \
  --db_root     "$DB_ROOT" \
  --final_iter  "$ITER" \
  --n_bins      "$NBINS" \
  --k_per_bin   "$KPERBIN" \
  --front                "$FRONT" \
  --ehvi_variant         "$EHVI_VARIANT" \
  --exploration_strategy "$EXPLORATION_STRATEGY" \
  --transform            "$TRANSFORM" \
  ${MC_EHVI:+--mc_ehvi} \
  ${LENGTH_CHANGES:+--length_changes}

echo "Endpoints have been generated, proceeding to beam searches..."

# 2) Run beams with MPI across all nodes
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
  --max_steps            60 \
  --tol_u                0.002 \
  --tol_v                0.002"

[[ "$MC_EHVI" == "true" ]]             && CMD+=" --mc_ehvi"
[[ "$LENGTH_CHANGES" == "true" ]]      && CMD+=" --length_changes"
[[ "$EXTEND_NO_FINISHED" == "true" ]]  && CMD+=" --extend_no_finished"
[[ "$EXTRA_STEPS" != "0" ]]            && CMD+=" --extra_steps $EXTRA_STEPS"
[[ "$STAGNATION_PATIENCE" != "0" ]]    && CMD+=" --stagnation_patience $STAGNATION_PATIENCE"
[[ "$STAGNATION_DELTA" != "0.0" ]]     && CMD+=" --stagnation_delta $STAGNATION_DELTA"

eval $CMD

echo "Beam searches complete, collecting results..."

# 3) Collect final master CSV
CMD="python collect_results.py \
  --scratch_dir $SCRATCH_AL \
  --model       $MODEL"
[[ "$LENGTH_CHANGES" == "true" ]] && CMD+=" --length_changes"
eval $CMD
