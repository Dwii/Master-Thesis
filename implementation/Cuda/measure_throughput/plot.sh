#!/bin/bash

gpu=pascal; python plot_benchmark_times_curve.py out/data_trsf_$gpu.pdf out/$gpu/cpu_to_gpu.dat "CPU vers GPU" out/$gpu/gpu_to_cpu.dat "GPU vers CPU"
gpu=titan;  python plot_benchmark_times_curve.py out/data_trsf_$gpu.pdf out/$gpu/cpu_to_gpu.dat "CPU vers GPU" out/$gpu/gpu_to_cpu.dat "GPU vers CPU"
gpu=tesla;  python plot_benchmark_times_curve.py out/data_trsf_$gpu.pdf out/$gpu/cpu_to_gpu.dat "CPU vers GPU" out/$gpu/gpu_to_cpu.dat "GPU vers CPU"

python plot_benchmark_times_curve.py out/bandwidth.pdf out/pascal/bandwidth.dat "Pascal" out/titan/bandwidth.dat "Titan" out/tesla/bandwidth.dat "Tesla"

