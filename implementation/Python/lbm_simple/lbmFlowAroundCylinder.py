#!/usr/bin/env python3
# Copyright (C) 2015 Universite de Geneve, Switzerland
# E-mail contact: jonas.latt@unige.ch
#
# 2D flow around a cylinder
#
# Update by Adrien Python (adrien.python@gmail.com):
# Slightly modified from original to handle different ordering for v/t variables

from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import datetime

###### Outputs definitions #####################################################
out = sys.argv[1]
outInterval = int(sys.argv[3])
outDir  = sys.argv[4]
outPre  = sys.argv[5]

###### Flow definition #########################################################
maxIter = int(sys.argv[2])  # Total number of time iterations.
Re = 220.0         # Reynolds number.
nx, ny = 420, 180 # Numer of lattice nodes.
ly = ny-1         # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.

###### Lattice Constants #######################################################

v = array([[ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
           [ 0, -1], [-1,  1], [-1,  0], [-1, -1]])
t = array([  1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36]) 

###### Automaticaly defined Constants ##########################################

def initcol(v0):
    return [f for f in range(9) if v[f][0] == v0]

col1 = initcol(1)
col2 = initcol(0)
col3 = initcol(-1)

def initopp():
    opp = [None] * 9
    for f in range(9):
        for g in range(9):
            if v[f][0] == -v[g][0] and  v[f][1] == -v[g][1]:
                opp[f] = g
                break
    return opp

opp  = initopp();

###### Function Definitions ####################################################
def macroscopic(fin):
    rho = sum(fin, axis=0)
    u = zeros((2, nx, ny))
    for i in range(9):
        u[0,:,:] += v[i,0] * fin[i,:,:]
        u[1,:,:] += v[i,1] * fin[i,:,:]
    u /= rho
    return rho, u

def equilibrium(rho, u):              # Equilibrium distribution function.
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    feq = zeros((9,nx,ny))
    for i in range(9):
        cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
# Creation of a mask with 1/0 values, defining the shape of the obstacle.
def obstacle_fun(x, y):
    return (x-cx)**2+(y-cy)**2<0 #(x-cx)**2+(y-cy)**2<r**2

obstacle = fromfunction(obstacle_fun, (nx,ny))

# Initial velocity profile: almost zero, with a slight perturbation to trigger
# the instability.
def inivel(d, x, y):
    return 0*y #(1-d) * uLB * (1 + 1e-4*sin(y/ly*2*pi))

vel = fromfunction(inivel, (2,nx,ny))

# Initialization of the populations at equilibrium with the given velocity.
rho = ones((nx,ny))
rho[nx//2,ny//2] = 2
fin = equilibrium(rho, vel)

seconds = 0
start = datetime.datetime.now()

###### Main time loop ##########################################################
for iter in range(1, maxIter+1):
    # Right wall: outflow condition.
#    fin[col3,-1,:] = fin[col3,-2,:] 

    # Compute macroscopic variables, density and velocity.
    rho, u = macroscopic(fin)

    # Left wall: inflow condition.
#    u[:,0,:] = vel[:,0,:]
#    rho[0,:] = 1/(1-u[0,0,:]) * ( sum(fin[col2,0,:], axis=0) +
#                                  2*sum(fin[col3,0,:], axis=0) )
    # Compute equilibrium.
    feq = equilibrium(rho, u)
#    for i in col1:
#        fin[i,0,:] = feq[i,0,:] + fin[opp[i],0,:] - feq[opp[i],0,:]

    # Collision step.
    fout = fin - omega * (fin - feq)

    # Bounce-back condition for obstacle.
    for i in range(9):
        fout[i, obstacle] = fin[opp[i], obstacle]

    # Streaming step.
    for i in range(9):
        fin[i,:,:] = roll(
                            roll(fout[i,:,:], v[i,0], axis=0),
                            v[i,1], axis=1 )
 
    if (outInterval==0 and iter == maxIter) or (outInterval>0 and iter % outInterval == 0):
        
        seconds += (datetime.datetime.now() - start).total_seconds()

        # Visualization of the velocity.
        if out == 'IMG':
            plt.clf()
            plt.imshow(sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
            plt.savefig("{0}/{1}{2}.png".format(outDir, outPre, iter))
        if out == 'OUT':
            file = open("{0}/{1}{2}.out".format(outDir, outPre, iter), 'w')
            for x in range(nx):
                for y in range(ny):
                    for f in range(9):
                        file.write("[{0}, {1}, {2}] {3:64.60f}\n".format(x,y,f, fin[f,x,y]))
        start = datetime.datetime.now()

print ("average lups:", nx*ny*maxIter/max(1,seconds))
