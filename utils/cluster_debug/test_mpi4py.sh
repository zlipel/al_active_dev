#!/bin/bash
#SBATCH --job-name=test_mpi4py
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=test_mpi4py.out
#SBATCH --error=test_mpi4py.err

module purge
module load openmpi/gcc/4.1.2
module load anaconda3/2024.6
conda activate torch-chemistry

srun python hello_mpi.py
