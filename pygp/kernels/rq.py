"""
Kernel which places a prior over periodic functions.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import RealKernel
from ._distances import sqdist, sqdist_per_dim
from ..utils.models import Printable

# exported symbols
__all__ = ['RQIso']


class RQIso(RealKernel, Printable):
    #---------------------------------------------------------------------------
    # Bookkeeping code

    def __init__(self, alpha, ell, sf, ndim=1):
        self._logalpha = np.log(float(alpha))
        self._logell = np.log(float(ell))
        self._logsf = np.log(float(sf))
        self.ndim = ndim
        self.nhyper = 3

    def _params(self):
        return (
            ('alpha', np.exp(self._logalpha)),
            ('ell', np.exp(self._logell)),
            ('sf', np.exp(self._logsf)),)

    def get_hyper(self):
        return np.r_[self._logalpha, self._logell, self._logsf]

    def set_hyper(self, hyper):
        self._logalpha = hyper[0]
        self._logell = hyper[1]
        self._logsf = hyper[2]

    #---------------------------------------------------------------------------
    # Kernel and gradient evaluation

    def get(self, X1, X2=None):
        alpha = np.exp(self._logalpha)
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)

        D = sqdist(ell, X1, X2)
        K = sf2 * (1 + 0.5*D/alpha) ** (-alpha)
        return K

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def grad(self, X1, X2=None):
        alpha = np.exp(self._logalpha)
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)

        D = sqdist(ell, X1, X2)
        E = 1 + 0.5*D/alpha
        K = sf2 * E**(-alpha)
        M = K*D/E

        yield 0.5*M - alpha*K*np.log(E)
        yield M
        yield 2*K

    def dgrad(self, X):
        yield np.zeros(len(X))
        yield np.zeros(len(X))
        yield 2 * self.dget(X)
