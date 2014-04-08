"""
Implementation of the squared-exponential kernels.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.spatial.distance as ssd
import abc

# local imports
from ..utils.models import Parameterized

# exported symbols
__all__ = ['Kernel', 'sqdist']


class Kernel(Parameterized):
    """
    Kernel interface.
    """
    @abc.abstractmethod
    def get(self): pass

    @abc.abstractmethod
    def dget(self): pass


def sqdist(ell, x1, x2=None):
    """
    Return the scaled squared-distance between two sets of vectors, x1 and x2,
    which should be passed as arrays of size (n,d) and (m,d) respectively. The
    vectors should be scaled by ell, which can be passed as either be a scalar
    or a d-vector.
    """
    if x2 is None:
        x2 = x1/ell
        return ssd.cdist(x2, x2, 'sqeuclidean')
    else:
        return ssd.cdist(x1/ell, x2/ell, 'sqeuclidean')
