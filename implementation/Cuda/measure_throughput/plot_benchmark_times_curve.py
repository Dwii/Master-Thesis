#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Display a list of *.dat files in a bar chart.
# Based on an example from https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html


from __future__ import unicode_literals
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import operator

FIRST_ARGS=2

if not (len(sys.argv) > FIRST_ARGS and (len(sys.argv)-FIRST_ARGS) % 2 == 0) :
    print("usage: python3 {0} <image path> [<dat1> <legend1>] .. [<datN> <legendN>] ) ".format(os.path.basename(sys.argv[0])))
    exit(1)

fig, ax = plt.subplots(figsize=(10,5))

image_path=sys.argv[1]
dats = int((len(sys.argv)-FIRST_ARGS)/2)

x_min = [None] * dats
x_max = [None] * dats
legends = [None] * dats

min_time = None
max_time = 0
time_by_lat = [None] * dats
for i, argi in enumerate(range(FIRST_ARGS, len(sys.argv), 2)):
    file  = sys.argv[argi]
    legends[i] = sys.argv[argi+1]
    time_by_lat[i] = ()
    # Load benchark
    for j, time in enumerate(open(file,'r')):
        time_by_lat[i] += ( [(j)**3*8/1E6, float(time)], )
        max_time = max(max_time, float(time))
        min_time = float(time) if min_time == None else min(min_time, float(time))

    x_min[i] = (min(time_by_lat[i], key=operator.itemgetter(0)))[0]
    x_max[i] = (max(time_by_lat[i], key=operator.itemgetter(0)))[0]


for i in range(dats):
    p=plt.plot(*zip(*time_by_lat[i]), label=legends[i])
#    plt.scatter(*zip(*time_by_lat[i]), color=p[0].get_color())

#plt.xticks(lat, domains)#, rotation='vertical')

#for i, label in enumerate(ax.xaxis.get_majorticklabels()):
#    if i == 1:
#        label.set_visible(False)

#plt.xticks(np.arange(0, max(x_max)+1, 1))
margin = max(min(x_min)*0.1, max(x_max)* 0.1)
ax.set_xlim(min(x_min)-margin, max(x_max)+margin)

ax.set_ylim(min_time, max_time * 1.2 )

# Set the chart's title
#ax.set_title(title)

ax.set_ylabel("Débit (Go/s)")
ax.set_xlabel('Données copiées (Mo)')

#fig.autofmt_xdate()
plt.legend(loc="upper left", ncol=dats)
#plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1), ncol=2)
plt.savefig(image_path, bbox_inches='tight')
#plt.show()
