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

    def __init__(self, sf, ell, alpha, ndim=1):
        self._logsf = np.log(float(sf))
        self._logell = np.log(float(ell))
        self._logalpha = np.log(float(alpha))
        self.ndim = ndim
        self.nhyper = 3

    def _params(self):
        return (
            ('sf', np.exp(self._logsf)),
            ('ell', np.exp(self._logell)),
            ('alpha', np.exp(self._logalpha)),
            )

    def get_hyper(self):
        return np.r_[self._logsf, self._logell, self._logalpha]

    def set_hyper(self, hyper):
        self._logsf = hyper[0]
        self._logell = hyper[1]
        self._logalpha = hyper[2]

    #---------------------------------------------------------------------------
    # Kernel and gradient evaluation

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        alpha = np.exp(self._logalpha)

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

        yield 2*K
        yield M
        yield 0.5*M - alpha*K*np.log(E)

    def dgrad(self, X):
        yield 2 * self.dget(X)
        yield np.zeros(len(X))
        yield np.zeros(len(X))
