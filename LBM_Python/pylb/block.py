import numpy as np
from scipy import weave
import cpp


class Block:
    def __init__(self, lattice, dim, omega, backend=None):
        self.lattice = lattice
        self.d = lattice.d
        self.dim = dim
        self.nx, self.ny, self.nz = dim if self.d == 3 else dim + (0,)
        self.omega = omega
        self.fin = self.equilibrium(1., np.zeros(dim + (self.d,)))
        # Bdvel and wall initialized to zero (but with proper dimensions).
        self.bdvel = np.zeros((0,)*(self.d+1))
        self.wall = np.zeros(dim, dtype=bool)
        self.force = np.zeros((0,)*(self.d+1))
        if backend is None:
            backend = cpp.Cpp2d if self.d == 2 else cpp.Cpp3d
        self._code = backend(lattice)
        self._cpp_collide_boundaries = self._code.collide_boundaries()
        self._cpp_inlet_outlet = self._code.inlet_outlet()
        self._cpp_collide_bulk_and_stream = self._code.collide_bulk_and_stream()

    def density(self):
        """ Calculates density on the whole domain. """
        rho = np.sum(self.fin, axis=self.d)
        # Define density-on-wall to be 1, to avoid division by zero.
        rho[self.wall] = 1
        return rho

    def velocity(self):
        """ Calculates velocity on the whole domain. """
        u = np.dot(self.fin, self.lattice.c)
        rho = self.density()
        for u_i in [self._component(u, i) for i in range(self.d)]:
            u_i /= rho
        # Define velocity-on-wall to be 0, for simpler post-processing.
        u[self.wall] = 0
        return u

    def wallforce(self):
        f = 2*np.dot(self.fin, self.lattice.c)
        f[~self.wall] = 0
        return f

    def velnorm(self):
        """ Calculates the norm of the velocity in the whole domain. """
        u = self.velocity()
        usqr = sum([self._component(u, i)**2 for i in range(self.d)])
        return np.sqrt(usqr)

    def equilibrium(self, rho, u):
        """ Calculates the equilibrium populations on the whole domain. """
        lat = self.lattice
        permutation = (range(self.d-1) + [self.d, self.d-1])
        cu = 3.0 * np.dot(lat.c, u.transpose(*permutation))
        usqr = sum([self._component(u, i)**2 for i in range(self.d)])
        usqr *= 3./2.
        feq = np.zeros(self.dim + (lat.q,))
        for i in range(lat.q):
            arg = [slice(None)]*self.d + [i]
            feq[arg] = rho * lat.t[i] * (1. + cu[i] + 0.5 * cu[i]**2 - usqr)
        return feq

    def inipopulations(self, vel):
        """ Initializes the fs at equilibrium with rho=1 and u=vel. """
        inivel = np.fromfunction(vel, self.dim + (self.d,)) \
                if callable(vel) else vel
        inirho = np.ones(self.dim)
        self.fin[~self.wall] = self.equilibrium(inirho, inivel)[~self.wall]

    def resetwall(self, vel):
        # Set populations on wall to zero, to make force calculation simpler.
        self.fin[self.wall] = 0;

    def setboundaryvel(self, vel):
        """ Specify velocity on inlet/outlet through a matrix or function. """
        # If vel is a function, create a nx-ny-2 or nx-ny-nz-3 matrix.
        self.bdvel = np.fromfunction(vel, self.dim + (self.d,)) \
                     if callable(vel) else vel

    def setforce(self, force):
        """ Specify body force through a matrix or function. """
        # If force is a function, create a nx-ny-2 or nx-ny-nz-3 matrix.
        self.force = np.fromfunction(force, self.dim + (self.d,)) \
                     if callable(force) else force

    def collide_boundaries(self):
        """ Execute collision step on outer domain boundaries. """
        self._inline(self._cpp_collide_boundaries)

    def inlet_outlet(self, has_inlet=True, has_outlet=True):
        """ Implement inlet and outlet condition. """
        # Determine if bd. condition is required on left and/or right boundary.
        if self.bdvel.size == 0:  # If user hasn't set any bd. condition...
            use_inlet, use_outlet = 0, 0
        else:
            use_inlet, use_outlet = int(has_inlet), int(has_outlet)
        self._inline(self._cpp_inlet_outlet,
                    {'use_inlet': use_inlet, 'use_outlet': use_outlet})

    def collide_bulk_and_stream(self):
        """ Execute collision step on bulk and streaming everywhere. """
        self._inline(self._cpp_collide_bulk_and_stream)

    def collide_and_stream(self):
        """ Execute collision and streaming everywhere. """
        self.collide_boundaries()
        self.collide_bulk_and_stream()
        self.inlet_outlet()

    def excess_algorithm(self):
        """ Return some algorithm to read/write boundary populations. """
        return CppExcess(self)

    def _component(self, u, i):
        """ Get slice into i-th comp. of an nx*ny*2 or nx*ny*nz*3 matrix. """
        return u[[slice(None)]*self.d + [i]];

    def _inline(self, cppCode, new_dict={}):
        """ Compile and execute a piece of C++ code. """
        global_dict = self.lattice._asdict()
        global_dict.update(new_dict)
        weave.inline( cppCode['code'], cppCode['var'],
                      local_dict=self.__dict__, global_dict=global_dict,
                      type_converters=weave.converters.blitz,
                      extra_compile_args=['-O3'])
                      #compiler='gcc', extra_compile_args=['-O3'])


class CppExcess:
    """ C++ implementation of read/write into outer-boundary populations. """
    def __init__(self, block):
        self.block = block
        self._cpp_num_excess = block._code.num_excess()
        self._cpp_put_excess = block._code.put_excess()
        self._cpp_get_excess = block._code.get_excess()
        self.numexcess = self.getnumexcess()
        self.ftmp = np.zeros(np.sum(self.numexcess))
        self.ofs = np.append(0, np.cumsum(self.numexcess))

    def getnumexcess(self):
        """ Number of excess pop. connected to each of the neighb. blocks. """
        d = self.block.lattice.d
        numexcess = np.zeros(3**d, dtype=int)
        self.block._inline(self._cpp_num_excess, {"numexcess": numexcess})
        return numexcess

    def getexcess(self):
        """ Copy excess variables from outer boundary into self.ftmp. """
        self.block._inline(self._cpp_get_excess, self.__dict__)

    def putexcess(self):
        """ Write back self.ftmp data into outer boundary. """
        self.block._inline(self._cpp_put_excess, self.__dict__)
