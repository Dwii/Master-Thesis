#!/bin/bash

N=$1

python3 plot_benchmark_times.py benchmarks/N$N.ben src_time_gpu_boundary_N$N.pdf "GPU N=$N (boundaries only)" 100 10 out/lbm_gpu_seq_boundary_N$N.dat "Palabos (temps restant)" out/lbm_gpu_seq_repeat-pc10_boundary_N$N.dat "Palabos Collide & Stream" out/lbm_gpu_seq_repeat-pd10_boundary_N$N.dat "Palabos Duplicate Overlaps" out/lbm_gpu_seq_repeat-s10_boundary_N$N.dat "coProcessor Send" out/lbm_gpu_seq_repeat-r10_boundary_N$N.dat "coProcessor Receive" out/lbm_gpu_seq_repeat-c10_boundary_N$N.dat "coProcessor Collide & Stream" 
python3 plot_benchmark_times.py benchmarks/N$N.ben src_time_cpu_boundary_N$N.pdf "CPU N=$N (boundaries only)" 100 10 out/lbm_cpu_seq_boundary_N$N.dat "Palabos (temps restant)" out/lbm_cpu_seq_repeat-pc10_boundary_N$N.dat "Palabos Collide & Stream" out/lbm_cpu_seq_repeat-pd10_boundary_N$N.dat "Palabos Duplicate Overlaps" out/lbm_cpu_seq_repeat-s10_boundary_N$N.dat "coProcessor Send" out/lbm_cpu_seq_repeat-r10_boundary_N$N.dat "coProcessor Receive" out/lbm_cpu_seq_repeat-c10_boundary_N$N.dat "coProcessor Collide & Stream" 
python3 plot_benchmark_times.py benchmarks/N$N.ben src_time_gpu_N$N.pdf          "GPU N=$N (full domain)"     100 10 out/lbm_gpu_seq_N$N.dat          "Palabos (temps restant)" out/lbm_gpu_seq_repeat-pc10_N$N.dat          "Palabos Collide & Stream" out/lbm_gpu_seq_repeat-pd10_N$N.dat          "Palabos Duplicate Overlaps" out/lbm_gpu_seq_repeat-s10_N$N.dat          "coProcessor Send" out/lbm_gpu_seq_repeat-r10_N$N.dat          "coProcessor Receive" out/lbm_gpu_seq_repeat-c10_N$N.dat          "coProcessor Collide & Stream" 
python3 plot_benchmark_times.py benchmarks/N$N.ben src_time_cpu_N$N.pdf          "CPU N=$N (full domain)"     100 10 out/lbm_cpu_seq_N$N.dat          "Palabos (temps restant)" out/lbm_cpu_seq_repeat-pc10_N$N.dat          "Palabos Collide & Stream" out/lbm_cpu_seq_repeat-pd10_N$N.dat          "Palabos Duplicate Overlaps" out/lbm_cpu_seq_repeat-s10_N$N.dat          "coProcessor Send" out/lbm_cpu_seq_repeat-r10_N$N.dat          "coProcessor Receive" out/lbm_cpu_seq_repeat-c10_N$N.dat          "coProcessor Collide & Stream" 