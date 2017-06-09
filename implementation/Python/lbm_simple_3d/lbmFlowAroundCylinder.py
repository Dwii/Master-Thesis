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
nx, ny, nz = 100, 100, 10 # Numer of lattice nodes.
ly = ny-1         # Height of the domain in lattice units.
cx, cy, cz, r = nx//4, ny//2, nz//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*10/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.

###### Lattice Constants #######################################################


v = array([[ 1,  1,  0], [ 1,  0,  0], [ 1, -1,  0], [ 0,  1,  0], [  0,  0,  0], [ 0, -1, 0], [-1,  1, 0], [-1,  0, 0], [-1, -1, 0],
           [ 1,  0,  1], [ 0,  1,  1], [ 0,  0,  1], [ 0, -1,  1], [ -1,  0,  1], 
           [ 1,  0, -1], [ 0,  1, -1], [ 0,  0, -1], [ 0, -1, -1], [ -1,  0, -1]])
t = array([  1./36, 1./18, 1./36, 1./18, 1./3 , 1./18, 1./36, 1./18, 1./36, 
             1./36, 1./36, 1./18, 1./36, 1./36, 
             1./36, 1./36, 1./18, 1./36, 1./36]) 

###### Automaticaly defined Constants ##########################################

def initcol(v0):
    return [f for f in range(19) if v[f][0] == v0]

col1 = initcol(1)
col2 = initcol(0)
col3 = initcol(-1)

def initopp():
    opp = [None] * 19
    for f in range(19):
        for g in range(19):
            if v[f][0] == -v[g][0] and v[f][1] == -v[g][1] and v[f][2] == -v[g][2]:
                opp[f] = g
                break
    return opp

opp  = initopp();

###### Function Definitions ####################################################
def macroscopic(fin):
    rho = sum(fin, axis=0)
    u = zeros((3, nx, ny, nz))
    for i in range(19):
        u[0,:,:,:] += v[i,0] * fin[i,:,:,:]
        u[1,:,:,:] += v[i,1] * fin[i,:,:,:]
        u[2,:,:,:] += v[i,2] * fin[i,:,:,:]
    u /= rho
    return rho, u

def equilibrium(rho, u):              # Equilibrium distribution function.
    usqr = 3/2 * (u[0]**2 + u[1]**2 + u[2]**2)
    feq = zeros((19,nx,ny,nz))
    for i in range(19):
        cu = 3 * (v[i,0]*u[0,:,:,:] + v[i,1]*u[1,:,:,:] + v[i,2]*u[2,:,:,:])
        feq[i,:,:,:] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
# Creation of a mask with 1/0 values, defining the shape of the obstacle.
def obstacle_fun(x, y, z):
    return (x-cx)**2+(y-cy)**2+(z-cz)**2<0 #(x-cx)**2+(y-cy)**2<r**2

obstacle = fromfunction(obstacle_fun, (nx,ny,nz))

# Initial velocity profile: almost zero, with a slight perturbation to trigger
# the instability.
def inivel(d, x, y, z):
    return 0*y #(1-d) * uLB * (1 + 1e-4*sin(y/ly*2*pi))

vel = fromfunction(inivel, (3,nx,ny,nz))

# Initialization of the populations at equilibrium with the given velocity.
rho = ones((nx,ny,nz))
rho[nx//2,ny//2,nz//2] = 2
fin = equilibrium(rho, vel)

seconds = 0
start = datetime.datetime.now()

if out == 'OUT':
    file = open("{0}/{1}{2}.out".format(outDir, outPre, 0), 'w')
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for f in range(19):
                    file.write("{0:64.60f}\n".format(fin[f,x,y,z]))


###### Main time loop ##########################################################
for iter in range(1, maxIter+1):

    # Compute macroscopic variables, density and velocity.
    rho, u = macroscopic(fin)

    # Compute equilibrium.
    feq = equilibrium(rho, u)

    # Collision step.
    fout = fin - omega * (fin - feq)

    # Bounce-back condition for obstacles
    for i in range(19):
        fout[i, obstacle] = fin[opp[i], obstacle]

    # Streaming step.
    for i in range(19):
        fin[i,:,:,:] =  roll(
                            roll(
                                roll(fout[i,:,:,:], v[i,0], axis=0),
                                v[i,1], axis=1 ),
                            v[i,2], axis=2 )

    if (outInterval==0 and iter == maxIter) or (outInterval>0 and iter % outInterval == 0):
        
        seconds += (datetime.datetime.now() - start).total_seconds()

        # Visualization of the velocity.
        if out == 'IMG':
            unorm_box = sqrt(u[0]**2+u[1]**2+u[2]**2).transpose()
            plt.clf()
            plt.imshow(unorm_box[nz//2,:,:], cmap=cm.Reds)
            plt.savefig("{0}/{1}{2}.png".format(outDir, outPre, iter))
        if out == 'OUT':
            file = open("{0}/{1}{2}.out".format(outDir, outPre, iter), 'w')
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        for f in range(19):
                            file.write("{0:64.60f}\n".format(fin[f,x,y,z]))
        start = datetime.datetime.now()

print ("average lups:", nx*ny*maxIter/max(1,seconds))
