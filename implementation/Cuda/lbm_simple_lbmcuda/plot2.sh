#!/bin/bash
# example usage: ./plot2.sh pascal benchmarks/opt.ben "$(seq 6)" 
out=out
ben=$2

function join_by { local IFS="$1"; shift; echo "$*"; };

for gpu in $1;
do
	args=''
	for line in $3;
	do
		files=$(ls $out | grep $gpu | grep -v repeat | grep -v sub)
		domain=$(sed "${line}q;d" $ben)
		domain=$(join_by x $domain)
		dat=${gpu}-${domain}.dat

		(for file in $files;
		do
			bs=$(echo $file | sed -e "s/lbm_${gpu}_B\([^_]*\)_.*/\1/")
			mlups=$(sed "${line}q;d" $out/$file)
			echo $bs $mlups
		done) | sort -k1 -g > $dat
		args="$args $dat '$domain'"
		dats="$dats $dat"
	done
	img=${gpu}_by_bs.pdf
	eval "python3 plot_tuple.py $img $args"
done
#rm $dats