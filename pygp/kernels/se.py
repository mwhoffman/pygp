"""
Implementation of the squared-exponential kernels.
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
__all__ = ['SEARD', 'SEIso']


class SEARD(RealKernel, Printable):
    def __init__(self, ell, sf):
        self._logell = np.log(np.ravel(ell))
        self._logsf = np.log(sf)
        self.nhyper = len(self._logell)+1

    def _params(self):
        return (
            ('ell', np.exp(self._logell)),
            ('sf', np.exp(self._logsf)),)

    def get_hyper(self):
        return np.r_[self._logell, self._logsf]

    def set_hyper(self, hyper):
        self._logell = hyper[:len(self._logell)]
        self._logsf  = hyper[-1]

    def get(self, X1, X2=None):
        D = sqdist(np.exp(self._logell), X1, X2)
        return np.exp(self._logsf*2 - 0.5*D)

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def grad(self, X1, X2=None):
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)
        for D in sqdist_per_dim(ell, X1, X2):
            D *= sf2 * np.exp(-0.5*D)
            yield D
        yield 2*self.get(X1, X2)

    def dgrad(self, X):
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)
        for i in xrange(len(self._logell)):
            yield np.zeros(len(X))
        yield 2 * sf2 * np.ones(len(X))

    def sample_spectrum(self, N):
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)
        W = np.random.randn(N, len(self._logell)) / ell
        return W, sf2


# NOTE: the definitions of the ARD and Iso kernels are basically the same, and
# will remain so until I implement the gradients. Trying to generalize this code
# seems to make the kernel a lot messier to read, so I'll leave it this way
# until I come up with a nicer way to do things.

class SEIso(RealKernel, Printable):
    def __init__(self, ell, sf, ndim):
        self._logell = np.log(float(ell))
        self._logsf = np.log(sf)
        self._ndim = ndim
        self.nhyper = 2

    def _params(self):
        return (
            ('ell', np.exp(self._logell)),
            ('sf', np.exp(self._logsf)),
            ('ndim', self._ndim),)

    def get_hyper(self):
        return np.r_[self._logell, self._logsf]

    def set_hyper(self, hyper):
        self._logell = hyper[0]
        self._logsf  = hyper[1]

    def get(self, X1, X2=None):
        D = sqdist(np.exp(self._logell), X1, X2)
        return np.exp(self._logsf*2 - 0.5*D)

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def grad(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        D = sqdist(ell, X1, X2)
        K = sf2 * np.exp(-0.5*D)
        D *= K
        yield D
        yield 2*K

    def dgrad(self, X):
        yield np.zeros(len(x1))
        yield 2*self.dget(X)

    def sample_spectrum(self, N):
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)
        W = np.random.randn(N, self._ndim) / ell
        return W, sf2
