# Display a list of *.dat files in a bar chart.
# Based on an example from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) > 3 and (len(sys.argv)-3) % 2 :
    print("usage: python3 {0} <benchmark> <image path> (<dat1> <legend1> [<dat2> <legend2>] .. [<datN> <legendN>] ) ".format(os.path.basename(sys.argv[0])))
    exit(1)

benchmark  = sys.argv[1]
image_path = sys.argv[2]
groups     = (len(sys.argv)-3)/2

# Load benchark
domains = ()
nb_setups = 0
for line in open(benchmark,'r'):
    n, snx, sny, snz = line.split()
    domains += ( r"{0}$^3$".format(snx), ) #+= ( "{0}x{1}x{2}".format(snx, sny, snz), )
    nb_setups += 1

# Setting the positions and width for the bars
pos = list(range(nb_setups)) 
width = 1 / (groups+2)

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

prop_iter = iter(plt.rcParams['axes.prop_cycle'])

legends = ()
maxLups = 0
for i, argi in enumerate(range(3, len(sys.argv), 2)):
    mlups = np.array(list(map(float, open(sys.argv[argi])))) / 1E6
    legends += ( sys.argv[argi+1], )
    maxLups = max(maxLups, max(mlups))
    plt.bar([p + width*i for p in pos], 
            mlups, 
            width, 
            alpha=0.5, 
            color=next(prop_iter)['color']) 

# Set the y axis label
ax.set_ylabel('MLUPS')
ax.set_xlabel('Taille du sous-domaine')

# Set the chart's title
#ax.set_title(title)

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(domains)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, maxLups] )

# Adding the legend and showing the plot
plt.legend(legends, loc='upper center')
ax.yaxis.grid()
plt.savefig(image_path)
plt.tight_layout()
plt.show()