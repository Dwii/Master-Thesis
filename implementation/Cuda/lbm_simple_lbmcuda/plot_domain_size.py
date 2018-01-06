
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
legends = [None] * dats

max_mlups = 0
for i, argi in enumerate(range(2, len(sys.argv), 2)):
    file  = sys.argv[argi]
    legends[i] = sys.argv[argi+1]
    mlups_by_ds = ()
    first = None
    last = None
    # Load benchark
    print(file)
    for j, lups in enumerate(open(file,'r')):
        mlups_by_ds += ( (int(j+30), int(lups) / 1E6), )
        max_mlups = max(max_mlups, int(lups) / 1E6)

    p=plt.plot(*zip(*mlups_by_ds), label=legends[i])

import math

xticks = list(range(32, 257, 16))
xticks_labels = [ r"{0}$^3$".format(tick) for tick in xticks]

plt.xticks(xticks, xticks_labels)
ax.set_xlim(16, 256+16)

ax.set_ylabel('MLUPS')
ax.set_xlabel('Taille du domaine')

plt.legend(loc="upper center", ncol=len(legends), prop={'size': 12} ) #, bbox_to_anchor=(0.5,-0.1)) 
ax.set_ylim(0,max_mlups*1.2)
plt.savefig(image_path, bbox_inches='tight')
plt.grid()
plt.show()
