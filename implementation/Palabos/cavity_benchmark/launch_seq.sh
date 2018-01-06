#!/bin/env bash
# Script to run with sbatch to execute benchmarks on slurm
# sbatch launch_mpi.sh

#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:titan:1

module load CUDA
module load GCC/4.9.3-2.25 OpenMPI/1.10.2
#module load icc/2017.1.132-GCC-6.3.0-2.27 impi/2017.1.132 Python/3.6.1


MPS=3
ITER=100
CHECK_NRG=true
RR=1
RS=1
RC=1
RPC=1
RPD=1

MPI=$(test $SLURM_TASKS_PER_NODE -eq 1 && echo true || echo false)
ARGS="ITER=$ITER MPS=$MPS CHECK_NRG=$CHECK_NRG MPIparallel=$MPI RR=$RR RS=$RS RC=$RC RPC=$RPC RPD=$RPD"

srun make benchmarks XPU=cpu BOUNDARY=false $ARGS
srun make benchmarks XPU=cpu BOUNDARY=true  $ARGS
srun make benchmarks XPU=gpu BOUNDARY=false $ARGS
srun make benchmarks XPU=gpu BOUNDARY=true  $ARGS