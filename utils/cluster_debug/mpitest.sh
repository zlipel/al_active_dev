#!/bin/bash
#SBATCH --job-name=beams_hps
#SBATCH --nodes=1                   # 4 nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node
#SBATCH --cpus-per-task=1      # 12 cores per rank
#SBATCH --time=00:29:59
#SBATCH --mem-per-cpu=10MB
#SBATCH --output=mpitest.out
#SBATCH --error=run_beams_hps.err

module purge

module load anaconda3/2024.6
module load openmpi/gcc/4.1.2
conda activate torch-chemistry

mpirun -V
