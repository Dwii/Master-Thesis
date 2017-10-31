# Display a list of *.dat files in a bar chart.
# Based on an example from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

FIRST_ARGS=6

if len(sys.argv) > FIRST_ARGS and (len(sys.argv)-FIRST_ARGS) % 2 :
    print("usage: python3 {0} <benchmark> <image path> <title> <iterations> <repeat> <dat1 (ref)> <legend1> [<dat2> <legend2>] .. [<datN> <legendN>]  ".format(os.path.basename(sys.argv[0])))
    exit(1)

benchmark  = sys.argv[1]
image_path = sys.argv[2]
title      = sys.argv[3]
iterations = int(sys.argv[4])
repeat     = int(sys.argv[5])
dats       = int((len(sys.argv)-6)/2)

# Load benchark
domains = ()
nb_setups = 0
for line in open(benchmark,'r'):
    n, snx, sny, snz = line.split()
    domains += ( "{0}x{1}x{2}".format(snx, sny, snz), )
    nb_setups += 1

# Setting the positions and width for the bars
pos = list(range(nb_setups)) 
width = 1 / (dats+2)

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

prop_iter = iter(plt.rcParams['axes.prop_cycle'])

legends = [None]*dats
bars = [None]*dats
maxLups = 0

#hatch_style = ("", "/" , "\\", "x", "o", "O", ".", "*" )
color = list(( next(prop_iter)['color'] for i in range(dats) ))

dats_mlups = [None]*dats
time = [None]*dats
maxTime=0
# Load mlups
for i, argi in enumerate(range(FIRST_ARGS, len(sys.argv), 2)):
    dats_mlups[i] = np.array(list(map(float, open(sys.argv[argi])))) / 1E6
    time[i] = (int(n)**3 * iterations) / dats_mlups[i] / 1000000
    legends[i] = sys.argv[argi+1]
    maxLups = max(maxLups, max(dats_mlups[i]))
    maxTime = max(maxTime, max(time[i]))

# Adjust mlups to get their delta to the reference
for i in range(1, len(dats_mlups)):
    dats_mlups[i] = abs(dats_mlups[0] - dats_mlups[i]) / (repeat-1)
    time[i] = abs(time[0] - time[i]) / (repeat-1)

bottom = 0
for i, mlups in reversed(list(enumerate(dats_mlups))):
    height = time[i] if i != 0 else abs(time[i]-bottom)
    bar = plt.bar([p + width for p in pos], height, width, bottom=bottom, alpha=0.5, color=color[i])
    bottom += time[i]
    bars[i] = bar
    
# Set the y axis label
ax.set_ylabel('Execution time (seconds)')

# Set the chart's title
ax.set_title(title)

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(domains)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, maxLups] )

# Adding the legend and showing the plot
lgd =plt.legend(bars, legends, loc='upper center', bbox_to_anchor=(0.5,-0.1))
#lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
ax.yaxis.grid()

#plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
#fig.autofmt_xdate()
plt.savefig(image_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.tight_layout()
#plt.show()