#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p azad
#SBATCH -t 24:30:00
#SBATCH -J force
#SBATCH -o force.o%j

module load gcc

srun -p azad -N 1 -n 1 -c 1  bash runAllFDGL.sh
