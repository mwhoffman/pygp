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
__all__ = ['sqdist', 'sqdist_per_dim']


def sqdist(ell, X1, X2=None):
    """
    Return the scaled squared-distance between two sets of vectors, x1 and x2,
    which should be passed as arrays of size (n,d) and (m,d) respectively. The
    vectors should be scaled by ell, which can be passed as either be a scalar
    or a d-vector.
    """
    if X2 is None:
        X1 = X1/ell
        X2 = X1
    else:
        X1 = X1/ell
        X2 = X2/ell
    return ssd.cdist(X1, X2, 'sqeuclidean')


def sqdist_per_dim(ell, X1, X2=None):
    if X2 is None:
        X1 = X1/ell
        X2 = X1
    else:
        X1 = X1/ell
        X2 = X2/ell
    for i in xrange(X1.shape[1]):
        yield ssd.cdist(X1[:,i,None], X2[:,i,None])
