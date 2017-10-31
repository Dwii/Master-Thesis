#!/bin/env bash
# Script to run with sbatch to execute benchmarks on slurm

#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:titan:1

module load CUDA
#module load icc/2017.1.132-GCC-6.3.0-2.27 impi/2017.1.132 Python/3.6.1
#module load GCC/4.9.3-2.25 OpenMPI/1.10.2

MPS=3
ITER=100
srun make benchmark ITER=$ITER XPU=cpu BOUNDARY=false MPS=$MPS
srun make benchmark ITER=$ITER XPU=cpu BOUNDARY=true  MPS=$MPS
srun make benchmark ITER=$ITER XPU=gpu BOUNDARY=false MPS=$MPS
srun make benchmark ITER=$ITER XPU=gpu BOUNDARY=true  MPS=$MPS

