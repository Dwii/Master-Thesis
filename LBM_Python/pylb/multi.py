import multiprocessing as mp
import lattice as mod_lattice
import block
import messages
import config as conf
import operator
from itertools import product
import numpy

def initialize(block, vel):
    block.inipopulations(vel)
    block.resetwall(vel)
    block.collide_and_stream()
    block.inipopulations(vel)

class BlockThread(block.Block, mp.Process):
    """Wraps up a block into a process, with which you'll communicate through
       a pipe, to access its interface through remote-procedure-calls."""
    def __init__(self, lattice, dim, omega, pipe):
        block.Block.__init__(self, lattice, dim, omega)
        mp.Process.__init__(self)
        self.nblock = pow(3, lattice.d)
        self.pipe = pipe
        # By default, the boundaries of the block are connected locally, in
        # a while that is compatible with periodic boundary conditions. If
        # instead the block is intended to be connected to neighbors, you must
        # reassign self.sender and self.recver.
        self.excess = self.excess_algorithm()
        comm = [messages.SerialPipe() for i in range(self.nblock)]
        self.sender = [send for send in comm]
        self.recver = [recv for recv in comm]

    def send(self):
        """ Send boundary data to neighboring blocks. """
        exc = self.excess
        exc.getexcess()
        for i in range(self.nblock):
            self.sender[i].send(exc.ftmp[exc.ofs[i]:exc.ofs[i+1]])

    def receive(self):
        """ Receive boundary data from neighboring blocks. """
        exc = self.excess
        for i in range(self.nblock):
           self.recver[i].recv(exc.ftmp[exc.ofs[i]:exc.ofs[i+1]])
        exc.putexcess()

    def run(self):
        # Start the remote-procedure-call loop.
        while True:
            message = self.pipe.recv()
            if message == "end":
                # Senders might need proper cleanup in the end.
                for s in self.sender:
                    try:
                        s.complete()
                    except AttributeError: pass
                break
            attr = getattr(self, message[0])
            if callable(attr):  # Case 1: attribute is a method.
                # Example: block.collide_boundaries(has_inlet, has_outlet)
                returnval = attr(*message[1])
                # Important: send only a return value if there's one, to avoid
                # performance penalties by unnecessarily synchronizing threads.
                if returnval is not None:
                    self.pipe.send(returnval)
            else:  # Case 2: attribute is a property.
                if message[1]:  # Setters have an argument: the rhs value.
                    # Setters. Example: block.wall = wallfun.
                    setattr(self, message[0], *message[1])
                else:
                    # Getters. Example: pop = block.fin.
                    self.pipe.send(attr)


class MultiBlock(object):
    def __init__(self, lattice, dim, omega, grid):
        self.q, self.d = lattice.q, lattice.d
        self.dim = dim
        self.omega = omega
        self.grid = grid
        self.my_blocks = [b for b in product(*[range(n) for n in self.grid])
                          if self.local(b)]
        for i in range(self.d):
            assert(dim[i]%grid[i]==0)
        self.bl_dim = tuple(dim[i] // grid[i] for i in range(self.d))
        self.comm = {bl: mp.Pipe() for bl in self.my_blocks}
        self.blocks = {bl: BlockThread( lattice, self.bl_dim,
                                        omega, self.comm[bl][1]) \
                       for bl in self.my_blocks}
        tag = 0
        self.recv_messages = dict()
        for bl in product(*[range(n) for n in self.grid]):
            for n in product(*[range(-1, 2) for i in range(self.d)]):
                self.register_communication(n, tag, bl)
                tag += 1
        for bl in self.my_blocks:
            self.blocks[bl].start()

    def register_communication(self, n, tag, bl):
        ngbor = tuple((bl[k] + n[k] + self.grid[k]) % self.grid[k] \
                      for k in range(self.d))
        side = self.linearindex([i+1 for i in n], [3]*self.d)
        bl_i = self.linearindex(bl, self.grid)
        ngbor_i = self.linearindex(ngbor, self.grid)
        if self.local(bl):
            size = self.blocks[bl].excess.numexcess[side]
            if bl == ngbor:
                comm = messages.SerialPipe()
                self.blocks[bl].sender[side] = comm
                self.blocks[ngbor].recver[side] = comm
            elif self.local(ngbor):
                #comm = messages.QueueAsPipe()
                comm = messages.ShmemPipe(size)
                #comm = messages.ZmqPipe("*", "jlatt-VirtualBox", tag)
                self.blocks[bl].sender[side] = comm
                self.blocks[ngbor].recver[side] = comm
            else:
               assert( False )  # Implement trans-processor communication here.
        elif self.local(ngbor):
            assert( False )

    def collide_and_stream(self):
        for bl in self.my_blocks:
            self.rpc(bl, "collide_boundaries")
            self.rpc(bl, "send")
        for bl in self.my_blocks:
            self.rpc(bl, "collide_bulk_and_stream")
            self.rpc(bl, "receive")
            self.rpc(bl, "inlet_outlet", bl[0]==0, bl[0]==self.grid[0]-1)

    def end(self):
        for bl in self.my_blocks:
            self.comm[bl][0].send("end")

    def onrank(self, grid_pos):
        i = self.linearindex(grid_pos, self.grid)
        return i // conf.thread_per_proc

    def local(self, grid_pos):
        return self.onrank(grid_pos) == conf.myrank

    def rpc(self, bl, method, *args):
        self.comm[bl][0].send([method, args])

    def rfc(self, bl, method, *args):
        self.rpc(bl, method, *args)
        return self.comm[bl][0].recv()

    def readvalue(self, attribute, ncomp, dtype):
        """Getter for a multi-block (e.g. "velnorm()"). Makes the RFC on each
           block, and assembles the result into the multi-block domain."""
        dim = self.dim + ((getattr(self, ncomp),) if ncomp else ())
        result = numpy.zeros(dim, dtype=dtype)
        for bl in self.my_blocks:
            self.sub(result, bl)[:] = self.rfc(bl, attribute)
        return result

    def setvalue(self, attribute, arg):
        """Setter for a multi-block (e.g. "inipopulations()"). For each block,
           extracts data from the multi-block and makes the RPC."""
        for bl in self.my_blocks:
            if callable(arg):
                localarg = self.localfun(arg, bl)
            else:
                localarg = self.sub(arg, bl).copy()
            self.rpc(bl, attribute, localarg)

    def ind(self, pos, dim, i):
        return pos[i] + dim[i] * self.ind(pos, dim, i-1) if i>0 else pos[0]

    def linearindex(self, pos, dim):
        return self.ind(pos, dim, self.d-1)

    def sub(self, mat, bl):
        return mat[[slice(i*n,(i+1)*n) for i, n in zip(bl, self.bl_dim)]]

    def localfun(self, fun, bpos):
        return SubWrap(fun, (bpos[i] * self.bl_dim[i] for i in range(self.d)))


class SubWrap(object):
    """Wrap a function to convert from multi-block to local coordinates."""
    def __init__(self, fun, pos):
        # Convert to tuple, because Python cannot serialize generators.
        self.fun, self.pos = fun, tuple(pos)

    def __call__(self, *arg):
        listarg = list(arg)
        for i, pos in enumerate(self.pos):
            listarg[i] += pos
        return self.fun(*listarg)


def multi_methods(cls):
    setters = {"inipopulations", "setboundaryvel", "setforce", "resetwall"}
    getters = {"density": "", "velocity": "d", "velnorm": "", "wallforce": "d"}
    properties = {"wall": ("", bool), "fin": ("q", float)}

    def set_method(name):
        def fun(self, arg):
            self.setvalue(name, arg)
        return fun

    def get_method(name, ncomp, dtype=float):
        def fun(self):
            return self.readvalue(name, ncomp, dtype)
        return fun

    for setter in setters:
        setattr(cls, setter, set_method(setter))
    for getter in getters:
        setattr(cls, getter, get_method(getter, getters[getter]))
    for prop in properties:
        fget = get_method(prop, properties[prop][0], properties[prop][1])
        fset = set_method(prop)
        setattr(cls, prop, property(fget=fget, fset=fset))

multi_methods(MultiBlock)


class GenerateBlock(object):
    def __init__(self, dim, omega, lattice=None, grid=None):
        if len(dim) == 2:
            lattice = lattice or getattr(mod_lattice, conf.Lattice2d)
            assert(conf.numproc * conf.thread_per_proc == \
                   reduce(operator.mul, conf.grid2d))
            grid = conf.grid2d
        elif len(dim) == 3:
            assert(conf.numproc * conf.thread_per_proc == \
                   reduce(operator.mul, conf.grid3d))
            lattice = lattice or getattr(mod_lattice, conf.Lattice3d)
            grid = conf.grid3d
        else:
            assert(False)
        if reduce(operator.mul, grid) == 1:
            self.block = block.Block(lattice, dim, omega)
        else:
            self.block = MultiBlock(lattice, dim, omega, grid)

    def __enter__(self):
        return self.block

    def __exit__(self, dtype, value, traceback):
        try:
            self.block.end()
        except AttributeError: pass
