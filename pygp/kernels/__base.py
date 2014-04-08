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
    def get(self, X1, X2=None):
        pass

    @abc.abstractmethod
    def dget(self, X):
        pass

    @abc.abstractmethod
    def transform(self, X):
        pass


class RealKernel(Kernel):
    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)


def sqdist(ell, X1, X2=None):
    """
    Return the scaled squared-distance between two sets of vectors, x1 and x2,
    which should be passed as arrays of size (n,d) and (m,d) respectively. The
    vectors should be scaled by ell, which can be passed as either be a scalar
    or a d-vector.
    """
    if X2 is None:
        X2 = X1/ell
        return ssd.cdist(X2, X2, 'sqeuclidean')
    else:
        return ssd.cdist(X1/ell, X2/ell, 'sqeuclidean')
