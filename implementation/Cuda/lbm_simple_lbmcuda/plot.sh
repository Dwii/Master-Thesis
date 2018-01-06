#!/bin/bash
# usage example: ./plot.sh tesla 2 benchmarks/tesla/opt.ben 


out=out
ben=$(basename $3 .ben)
r=10

for gpu in $1;
do
	for bs in $2;
	do
		python3 plot_benchmark_times.py       $3 $gpu-$ben-$bs.pdf "$ben.ben Benchmark on $gpu GPU with $bs block size"       100 $r $out/lbm_${gpu}_B${bs}_$ben.dat "Temps restant" $out/lbm_${gpu}_B${bs}__repeat-s$r$ben.dat "Send" $out/lbm_${gpu}_B${bs}__repeat-r$r$ben.dat "Receive" $out/lbm_${gpu}_B${bs}__repeat-c$r$ben.dat "Collide & Stream" 
		python3 plot_benchmark_times_curve.py $3 $gpu-$ben-$bs.pdf "Temps d'execution sur un GPU $gpu (blocs de $bs threads)" 100 $r $out/lbm_${gpu}_B${bs}_$ben.dat "Temps total"   $out/lbm_${gpu}_B${bs}__repeat-s$r$ben.dat "Send" $out/lbm_${gpu}_B${bs}__repeat-r$r$ben.dat "Receive" $out/lbm_${gpu}_B${bs}__repeat-c$r$ben.dat "Collide & Stream" 
	done
done