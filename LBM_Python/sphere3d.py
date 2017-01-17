# Copyright (C) 2013 FlowKit Ltd
from numpy import *
from pylb import multi
from pylb import lbio

#def inivelfun(x, y, z, d):
#    """ v_x(x,y) = uMax*(1+.2*sin(y/ly*2pi)+.2*sin(z/lz*2pi)). v_y(x,y) = v_y(x,y)= 0 """
#    return (d==0) * uLB * (1.0 + 1e-2 * sin(y/ly *2*pi) +
#                                 1e-2 * sin(z/lz *2*pi))


class InivelFun(object):
    def __init__(self, uLB, ly, lz):
        self.uLB, self.ly, self.lz = uLB, ly, lz

    def __call__(self, x, y, z, d):
        """ v_x(x,y) = uMax*(1+.2*sin(y/ly*2pi)+.2*sin(z/lz*2pi)). v_y(x,y) = v_y(x,y)= 0 """
        return (d==0) * self.uLB * (1.0 + 1e-2 * sin(y/self.ly *2*pi) +
                                          1e-2 * sin(z/self.lz *2*pi))

def cylinder(nx=160, ny=60, nz=60, Re=220.0, maxIter=10000, plotImages=True):
    ly=ny-1.0
    lz=nz-1.0
    cx, cy, cz = nx/4, ny/2, nz/2
    r=ny/9   # Coordinates of the cylinder.
    uLB     = 0.04                # Velocity in lattice units.
    nulb = uLB * r / Re
    omega = 1.0 / (3. * nulb + 0.5); # Relaxation parameter.

    with multi.GenerateBlock((nx, ny, nz), omega) as block:
        block.wall = fromfunction(lambda x, y, z: (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < r**2, (nx, ny, nz))
        inivelfun = InivelFun(uLB, ly, lz)
        inivel = fromfunction(inivelfun, (nx, ny, nz, 3))
        block.inipopulations(inivelfun)
        block.setboundaryvel(inivelfun)

        if plotImages:
            plot = lbio.Plot(block.velnorm()[:,:,nz//2])

        for time in range(maxIter):
            block.collide_and_stream()

            if (plotImages and time%10==0):
                lbio.writelog(sum(sum(sum(block.wallforce()[:,:,:,0]))))
                plot.draw(block.velnorm()[:,:,nz//2])
                #print(block.fin[10,10,10,3])
                #plot.savefig("vel."+str(time/100).zfill(4)+".png")


if __name__ == "__main__":
    cylinder(maxIter=10000, plotImages=True)
