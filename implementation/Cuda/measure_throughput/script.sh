#!/bin/env bash

#SBATCH --partition=shared-gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:tesla:1

module load CUDA

gpu=tesla
dir=out/$gpu
bin=bin
mkdir -p $dir $bin
#nvcc measure_effective_bandwidth.cu -O3 -o $bin/measure_effective_bandwidth
#nvcc measure_cudamemcpy_to_gpu.cu -O3 -o $bin/measure_cudamemcpy_to_gpu
#nvcc measure_cudamemcpy_to_cpu.cu -O3 -o $bin/measure_cudamemcpy_to_cpu

for i in $(seq 1 256); 
do
  echo "N = $i^3"
  n=$(($i ** 3))
  srun $bin/measure_effective_bandwidth $n >> $dir/bandwidth.dat
  srun $bin/measure_cudamemcpy_to_gpu $n >> $dir/cpu_to_gpu.dat
  srun $bin/measure_cudamemcpy_to_cpu $n >> $dir/gpu_to_cpu.dat
done
