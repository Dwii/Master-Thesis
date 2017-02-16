#!/usr/bin/env python3
#
# Compare two lists of floats and give their average delta.
#
import sys
import os.path

if len(sys.argv) != 3:
	print("usage: python3 {0} <file1> <file2>".format(os.path.basename(sys.argv[0])))
	exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

with open(file1) as f:
    results1 = list(map(float, f))

with open(file2) as f:
    results2 = list(map(float, f))

maxi = max(len(results1), len(results2))
mini = min(len(results1), len(results2))

if mini != maxi:
	print("amount of results differ")
	exit(0)

deltas = [0] * mini
for i in range(mini):
	deltas[i] = abs(results1[i] - results2[i])

delta = abs(sum(deltas) / mini)

print("{0:.60f}".format(delta))
