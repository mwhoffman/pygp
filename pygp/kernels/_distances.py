"""
Implementation of distance computations.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import scipy.spatial.distance as ssd

# exported symbols
__all__ = ['rescale', 'sqdist', 'sqdist_foreach']


def rescale(ell, X1, X2):
    """
    Rescale the two sets of vectors by `ell`.
    """
    X1 = X1 / ell
    X2 = X2 / ell if (X2 is not None) else None
    return X1, X2


def diff(X1, X2=None):
    """
    Return the differences between vectors in `X1` and `X2`. If `X2` is not
    given this will return the pairwise differences in `X1`.
    """
    X2 = X1 if (X2 is None) else X2
    return X1[:, None, :] - X2[None, :, :]


def sqdist(X1, X2=None):
    """
    Return the squared-distance between two sets of vector. If `X2` is not
    given this will return the pairwise squared-distances in `X1`.
    """
    X2 = X1 if (X2 is None) else X2
    return ssd.cdist(X1, X2, 'sqeuclidean')


def sqdist_foreach(X1, X2=None):
    """
    Return an iterator over each dimension returning the squared-distance
    between two sets of vector. If `X2` is not given this will iterate over the
    pairwise squared-distances in `X1` in each dimension.
    """
    X2 = X1 if (X2 is None) else X2
    for i in xrange(X1.shape[1]):
        yield ssd.cdist(X1[:, i, None], X2[:, i, None], 'sqeuclidean')
