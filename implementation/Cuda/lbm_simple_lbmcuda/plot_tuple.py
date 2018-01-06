# Display a list of *.dat files in a bar chart.
# Based on an example from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import operator

if not (len(sys.argv) > 2 and (len(sys.argv)-2) % 2 == 0) :
    print("usage: python3 {0} <image path> <dat1> <legend1> [<dat2> <legend2>] .. [<datN> <legendN>] ) ".format(os.path.basename(sys.argv[0])))
    exit(1)

fig, ax = plt.subplots(figsize=(10,5))

image_path=sys.argv[1]
dats = int(len(sys.argv)/2)-1
x_min = [None] * dats
x_max = [None] * dats
legends = [None] * dats

def get_legend(domain):
    domain_set = set(domain.split('x'))
    if len(domain_set) == 1:
        legend = r"{0}$^3$".format(domain_set.pop())
    else:
        legend = domain
    return legend

max_mlups = 0
for i, argi in enumerate(range(2, len(sys.argv), 2)):
    file  = sys.argv[argi]
    legends[i] = get_legend(sys.argv[argi+1])
    mlups_by_bs = ()
    first = None
    last = None
    # Load benchark
    for line in open(file,'r'):
        bs, lups = line.split()
        mlups_by_bs += ( (int(bs), int(lups) / 1E6), )
        max_mlups = max(max_mlups, int(lups) / 1E6)

    x_min[i] = (min(mlups_by_bs, key=operator.itemgetter(0)))[0]
    x_max[i] = (max(mlups_by_bs, key=operator.itemgetter(0)))[0]

    p=plt.plot(*zip(*mlups_by_bs), label=legends[i])
    plt.scatter(*zip(*mlups_by_bs), color=p[0].get_color())

import math

#xticks = [tick for tick in np.arange(min(x_min), max(x_max)+1, 4)]
max_p2 = min(16, max(x_max))
xticks = [2**i for i in range(1, int(math.log2(max_p2))+1) ] + [i for i in range(max_p2, max(x_max)+1, 16) ]

plt.xticks(xticks)
ax.set_xlim(min(x_min)-1, max(x_max)+1)

ax.set_ylabel('MLUPS')
ax.set_xlabel('Taille des blocs')

plt.legend(title="Taille du domaine",loc="upper center", ncol=len(legends), prop={'size': 12} ) #, bbox_to_anchor=(0.5,-0.1)) 
ax.set_ylim(0,max_mlups*1.25)
plt.savefig(image_path, bbox_inches='tight')
plt.show()
