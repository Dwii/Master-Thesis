# Display a list of *.dat files in a bar chart.
# Based on an example from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import operator

FIRST_ARGS=6

if not (len(sys.argv) > FIRST_ARGS and (len(sys.argv)-FIRST_ARGS) % 2 == 0) :
    print("usage: python3 {0} <benchmark> <image path> <title> <iterations> <repeat> <dat1> <legend1> [<dat2> <legend2>] .. [<datN> <legendN>] ) ".format(os.path.basename(sys.argv[0])))
    exit(1)

fig, ax = plt.subplots(figsize=(10,5))

benchmark  = sys.argv[1]
image_path=sys.argv[2]
title = sys.argv[3]
iterations = int(sys.argv[4])
repeat = int(sys.argv[5])

dats = int((len(sys.argv)-FIRST_ARGS)/2)

x_min = [None] * dats
x_max = [None] * dats
legends = [None] * dats

domains = ()
nb_setups = 0
lat = ()
for line in open(benchmark,'r'):
    nx, ny, nz = line.split()
    lat += (int(nx)*int(ny)*int(nz),)
    domains += ( r"{0}$^3$".format(nx), )
    nb_setups += 1


time_by_lat = [None] * dats
for i, argi in enumerate(range(FIRST_ARGS, len(sys.argv), 2)):
    file  = sys.argv[argi]
    legends[i] = sys.argv[argi+1]
    time_by_lat[i] = ()
    first = None
    last = None
    # Load benchark
    bs= 1

    for j, lups in enumerate(open(file,'r')):
        time = (lat[j] * iterations) / ( int(lups) )
        time_by_lat[i] += ( [lat[j], time], )


    x_min[i] = (min(time_by_lat[i], key=operator.itemgetter(0)))[0]
    x_max[i] = (max(time_by_lat[i], key=operator.itemgetter(0)))[0]


# Adjust mlups to get their delta to the reference
for i in range(1, len(time_by_lat)):
    for j in range(len(time_by_lat[i])):
        time_by_lat[i][j][1] = abs(time_by_lat[0][j][1] - time_by_lat[i][j][1]) / (repeat-1)

for i in range(len(time_by_lat)):
    p=plt.plot(*zip(*time_by_lat[i]), label=legends[i])
    plt.scatter(*zip(*time_by_lat[i]), color=p[0].get_color())

plt.xticks(lat, domains)#, rotation='vertical')

for i, label in enumerate(ax.xaxis.get_majorticklabels()):
    if i == 1:
        label.set_visible(False)

#plt.xticks(np.arange(0, max(x_max)+1, 1))
margin = max(min(x_min)*0.1, max(x_max)* 0.1)
ax.set_xlim(min(x_min)-margin, max(x_max)+margin)

ax.set_ylim(0, (max(time_by_lat[0], key=operator.itemgetter(0)))[1] * 1.1 )

# Set the chart's title
#ax.set_title(title)

ax.set_ylabel("Temps d'execution (secondes)")
ax.set_xlabel('Taille du domaine')

#fig.autofmt_xdate()
plt.legend(loc="upper left",)
#plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1), ncol=2)
plt.savefig(image_path, bbox_inches='tight')
#plt.show()
