#!/usr/bin/env python3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations

NX=4
NY=4
NZ=4

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def IDX(x, y, z, nx=NX, ny=NY, nz=NZ):
# Copy paste IDX macro after return:
	return ((x+(nx))%(nx) + ((z+(nz))%(nz) + ( (y+(ny))%(ny) )*(nz))*(nx))

def cube(x,y,z,l, label='', sides=True, frame=True, alpha=1):
	# Cube sides
	if sides:
		sides_alpha = alpha*0.3
		btX = np.array([[x,x+l],[x,x+l]])
		btY = np.array([[y,y],[y+l,y+l]])
		frX = np.array([[x,x+l],[x,x+l]])
		frZ = np.array([[z,z],[z+l,z+l]])
		lrY = np.array([[y,y+l],[y,y+l]])
		lrZ = np.array([[z,z],[z+l,z+l]])
	
		ax.plot_surface(btX, btY, z+l, alpha=sides_alpha, color='white')
		ax.plot_surface(btX, btY,   z, alpha=sides_alpha, color='white')
		ax.plot_surface(frX,   y, frZ, alpha=sides_alpha, color='white')
		ax.plot_surface(frX, y+l, frZ, alpha=sides_alpha, color='white')
		ax.plot_surface(x+l, lrY, lrZ, alpha=sides_alpha, color='white')
		ax.plot_surface(  x, lrY, lrZ, alpha=sides_alpha, color='white')
	
	# Cube frame
	if frame:
		X = [x, x+1]
		Y = [y, y+1]
		Z = [z, z+1]
		for s, e in combinations(np.array(list(product(X, Y, Z))), 2):
			if np.sum(np.abs(s-e)) == X[1]-X[0]:
				ax.plot3D(*zip(s, e), color="b", alpha=alpha)

	# Cube label
	center_diff=l/2
	ax.text(x+center_diff, y+center_diff, z+center_diff, label, None, alpha=alpha, color='red')

### Visualisaion

space=0.5

for x in range(NX):
	for y in range(NY):
		for z in range(NZ):
			cx = x+x*space
			cy = y+y*space
			cz = z+z*space
			axis = x # Set axis to emphasize here
			alpha = 1 if axis == 0 else 0.1
			label = '%d' % IDX(x,y,z)
			cube(cx,cy,cz,1, label=label, alpha=alpha)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
