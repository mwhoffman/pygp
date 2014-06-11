"""
Implementation of distance computations.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.spatial.distance as ssd

# exported symbols
__all__ = ['sqdist']


def sqdist(X1, X2=None):
    """
    Return the scaled squared-distance between two sets of vectors, x1 and x2,
    which should be passed as arrays of size (n,d) and (m,d) respectively. The
    vectors should be scaled by ell, which can be passed as either be a scalar
    or a d-vector.
    """
    X2 = X1 if (X2 is None) else X2
    return ssd.cdist(X1, X2, 'sqeuclidean')


def dist(X1, X2=None):
    return np.sqrt(sqdist(X1, X2))
