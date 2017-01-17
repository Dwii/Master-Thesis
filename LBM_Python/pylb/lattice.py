'''
This module defines the structure of the most commonly used
2D and 3D lattices.
'''

import numpy as np
import collections
from itertools import product


Lattice = collections.namedtuple('Lattice', ('d', 'q', 'c', 't'))


def _lattice2d(q, weights):
    '''Generates a 2D lattice.

    The weights must be provided as a dictionary with the Manhattan length of
    the velocities as the key. Velocities of a length missing from the
    dictionary will not be generated.
    '''
    manhattan = lambda x, y: abs(x) + abs(y)
    c = np.array([ci for ci in product([-1, 0, 1], [-1, 0, 1])
                  if manhattan(*ci) in weights])
    t = np.array([weights[manhattan(*ci)] for ci in c])
    return Lattice(2, q, c, t)


def _lattice3d(q, weights):
    '''Generates a 2D lattice. Same convection as _lattice2d.'''
    manhattan = lambda x, y, z: abs(x) + abs(y) + abs(z)
    c = np.array([ci for ci in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
                  if manhattan(*ci) in weights])
    t = np.array([weights[manhattan(*ci)] for ci in c])
    return Lattice(3, q, c, t)


# D2Q9 contains all velocities.
D2Q9 = _lattice2d(q=9, weights={0: 4./9.,
                                1: 1./9.,
                                2: 1./36.})

# D3Q19 has no velocities of length sqrt(3).
D3Q19 = _lattice3d(q=19, weights={0: 1./3.,
                                  1: 1./18.,
                                  2: 1./36.})

# D3Q15 has no velocities of length sqrt(2).
D3Q15 = _lattice3d(q=15, weights={0: 2./9.,
                                  1: 1./9.,
                                  3: 1./72.})

# D3Q27 contains all velocities.
D3Q27 = _lattice3d(q=27, weights={0: 8./27.,
                                  1: 2./27.,
                                  2: 1./54.,
                                  3: 1./216.})
