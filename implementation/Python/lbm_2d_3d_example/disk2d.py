# Copyright (C) 2013 FlowKit Ltd
from numpy import *
from pylb import multi
from pylb import lbio

#def inivelfun(x, y, d):
#    """ v_x(x,y) = uMax*(1+.2*sin(y/ly*2pi)). v_y(x,y) = 0 """
#    return (1-d) * uLB * (1.0 + 1e-2 * sin(y/ly * 2*pi))


class InivelFun(object):
    def __init__(self, uLB, ly):
        self.uLB, self.ly = uLB, ly

    def __call__(self, x, y, d):
        """ v_x(x,y) = uMax*(1+.2*sin(y/ly*2pi)). v_y(x,y) = 0 """
        return (1-d) * self.uLB * (1.0 + 1e-2 * sin(y/self.ly * 2*pi))


def cylinder(nx=2000, ny=1200, Re=220.0, maxiter=10000, plotimages=True):
    ly=ny-1.0
    cx, cy = nx // 4, ny // 2  # Coordinates of the cylinder.
    r = ny // 20;
    uLB = 0.05  # Velocity in lattice units.
    nulb = uLB * r / Re
    omega = 1.0 / (3.*nulb + 0.5)  # Relaxation parameter.
    image_freq = 50

    with multi.GenerateBlock((nx, ny), omega) as block:
        block.wall = fromfunction(lambda x, y: (x-cx)**2 + (y-cy)**2 < r**2, (nx, ny))
        inivelfun = InivelFun(uLB, ly)
        multi.initialize(block, inivelfun)
        block.setboundaryvel(inivelfun)

        if plotimages:
            plot = lbio.Plot(block.velnorm())

        for time in range(maxiter):
            block.collide_and_stream()

            if (plotimages and time%image_freq==0):
                plot.draw(block.velnorm())
                #plot.draw(block.wallforce()[:,:,0])
                cd = sum(sum(block.wallforce()[:,:,0])) / (uLB**2*r)
                fluid = ~block.wall
                #fluid = ~block.wall.astype(bool)
                rhoAve = sum(block.density()[fluid])/sum(sum(fluid))
                lbio.writelog("CD={0}, rhoAve={1}".format(cd, rhoAve))
                lbio.writelog(sum(sum(block.wallforce()[:,:,0])) / (uLB**2*r))
                lbio.save("velocity.txt", block.velnorm())
                #lbio.writelog(block.fin[10,10,3])
                plot.savefig("vel."+str(time/image_freq).zfill(4)+".png")


if __name__ == "__main__":
    cylinder(Re=1000, maxiter=500000, plotimages=True)

