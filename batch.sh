#!/bin/sh
#SBATCH --job-name=bootstrap
#SBATCH --partition=lq1_cpu
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40

module load mambaforge
conda activate default

srun --ntasks 1 --cpus-per-task 40 python -u bootstrap.py
