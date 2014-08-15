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
from ..utils.models import printable
from ._distances import rescale, sqdist, sqdist_foreach
from ._base import RealKernel

# exported symbols
__all__ = ['RQ']


@printable
class RQ(RealKernel):
    def __init__(self, sf, ell, alpha, ndim=None):
        self._logsf = np.log(float(sf))
        self._logell = np.log(ell)
        self._logalpha = np.log(float(alpha))

        self._iso = False
        self.ndim = np.size(self._logell)
        self.nhyper = 2 + np.size(self._logell)

        if ndim is not None:
            if np.size(self._logell) == 1:
                self._logell = float(self._logell)
                self._iso = True
                self.ndim = ndim
            else:
                raise ValueError('ndim only usable with scalar lengthscales')

    def _params(self):
        return [
            ('sf', 1),
            ('ell', self.nhyper-2),
            ('alpha', 1),
        ]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell, self._logalpha]

    def set_hyper(self, hyper):
        self._logsf = hyper[0]
        self._logell = hyper[1] if self._iso else hyper[1:-1]
        self._logalpha = hyper[-1]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        alpha = np.exp(self._logalpha)

        X1, X2 = rescale(ell, X1, X2)
        K = sf2 * (1 + 0.5*sqdist(X1, X2)/alpha) ** (-alpha)
        return K

    def grad(self, X1, X2=None):
        # hypers
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        alpha = np.exp(self._logalpha)

        # precomputations
        X1, X2 = rescale(ell, X1, X2)
        D = sqdist(X1, X2)
        E = 1 + 0.5*D/alpha
        K = sf2 * E**(-alpha)
        M = K*D/E

        yield 2*K                               # derivative wrt logsf
        if self._iso:
            yield M                             # derivative wrt logell (iso)
        else:
            for D in sqdist_foreach(X1, X2):
                yield K*D/E                     # derivative wrt logell (ard)
        yield 0.5*M - alpha*K*np.log(E)         # derivative wrt alpha

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def dgrad(self, X):
        yield 2 * self.dget(X)
        for _ in xrange(self.nhyper-2):
            yield np.zeros(len(X))
        yield np.zeros(len(X))
