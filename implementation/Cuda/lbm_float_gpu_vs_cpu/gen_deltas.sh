#!/bin/bash

ITER=$1
OUTDIR=out
TOOLDIR=../../Tools
GPUPRE=gpu_lbm_
CPUPRE=cpu_lbm_

size=44226000 # NX * NY * F * Print digits (64+the dot) = 420*180*9*65

printf "" > $OUTDIR/deltas.csv

for iter in $(seq 5470 1 $ITER); do 
	gpu_out=$OUTDIR/$GPUPRE$iter.out 
	cpu_out=$OUTDIR/$CPUPRE$iter.out 

	while [[ ! -s $gpu_out || ! -s $cpu_out || $(cat $gpu_out | wc -c) -ne $size || $(cat $cpu_out | wc -c) -ne $size ]]
	do 
		echo "wait for $gpu_out and $cpu_out"
		sleep 2
	done

	delta=$(python3 $TOOLDIR/floats_delta.py $gpu_out $cpu_out)
	printf "$iter;$delta\n" >> $OUTDIR/deltas.csv
	rm $gpu_out $cpu_out
done

