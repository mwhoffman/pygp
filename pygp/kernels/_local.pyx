"""
Wrapper around the covariance computation code for ES/EP.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['local_se']

# external symbols
cdef extern from *:
    void se_cov (
        double *K, double *X, int n, int d, double *x, double *ell, double sf2,
        double sn2)

    void se_crosscov (
        double *K, double *X1, int n, int d, double *x, double *ell, double sf2,
        double sn2, double *X2, int m)


def local_se(double[:] ell,
             double sf2,
             double sn2,
             double[:] x,
             double[::1, :] X1,
             double[::1, :] X2=None):

    # get the sizes we'll need
    cdef int n = X1.shape[0]
    cdef int d = X1.shape[1]
    cdef int m
    cdef int r
    cdef double[::1, :] K

    if X2 is None:
        r = n + d*d + d + 1
        K = np.empty((r, r), order='F')
        se_cov(&K[0,0], &X1[0,0], n, d, &x[0], &ell[0], sf2, sn2)

    else:
        m = X2.shape[0]
        r = n + d + int(d*(d-1)/2) + d + 1
        K = np.empty((m, r), order='F')
        se_crosscov(&K[0,0], &X1[0,0], n, d, &x[0], &ell[0], sf2, sn2, &X2[0,0], m)

    return np.asarray(K)
