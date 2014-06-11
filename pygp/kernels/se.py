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
from ._distances import sqdist
from ._local import local_se

from ..utils.random import rstate
from ..utils.models import Printable

# exported symbols
__all__ = ['SEARD', 'SEIso']


class SEARD(RealKernel, Printable):
    def __init__(self, sf, ell):
        self._logsf = np.log(float(sf))
        self._logell = np.log(np.ravel(ell))
        self.ndim = len(self._logell)
        self.nhyper = self.ndim + 1

    def _params(self):
        return [
            ('sf',  1),
            ('ell', self.ndim),]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf  = hyper[0]
        self._logell = hyper[1:]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        X1 = X1 / ell
        X2 = X2 / ell if (X2 is not None) else None
        K = sf2 * np.exp(-0.5*sqdist(X1, X2))
        return K

    def grad(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        X1 = X1 / ell
        X2 = X2 / ell if (X2 is not None) else None
        K = sf2 * np.exp(-0.5*sqdist(X1, X2))
        yield 2*K
        for i in xrange(self.ndim):
            D = sqdist(X1[:,i,None], X2[:,i,None] if (X2 is not None) else None)
            yield K*D

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def dgrad(self, X):
        yield 2 * np.exp(self._logsf*2) * np.ones(len(X))
        for i in xrange(self.ndim):
            yield np.zeros(len(X))

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        W = rng.randn(N, self.ndim) / ell
        return W, sf2

    def get_local(self, x, X1, X2=None):
        ell = np.exp(self._logell*-2)
        sf2 = np.exp(self._logsf*2)
        return local_se(ell, sf2, 0.0, x, X1, X2)


# NOTE: the definitions of the ARD and Iso kernels are basically the same, and
# will remain so until I implement the gradients. Trying to generalize this code
# seems to make the kernel a lot messier to read, so I'll leave it this way
# until I come up with a nicer way to do things.

class SEIso(RealKernel, Printable):
    def __init__(self, sf, ell, ndim=1):
        self._logsf  = np.log(float(sf))
        self._logell = np.log(float(ell))
        self.ndim = ndim
        self.nhyper = 2

    def _params(self):
        return [
            ('sf',  1),
            ('ell', 1),]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf  = hyper[0]
        self._logell = hyper[1]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        X1 = X1 / ell
        X2 = X2 / ell if (X2 is not None) else None
        K = sf2 * np.exp(-0.5*sqdist(X1, X2))
        return K

    def grad(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        X1 = X1 / ell
        X2 = X2 / ell if (X2 is not None) else None
        D = sqdist(X1, X2)
        K = sf2 * np.exp(-0.5*D)
        yield 2*K
        yield K*D

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def dgrad(self, X):
        yield 2 * np.exp(self._logsf*2) * np.ones(len(X))
        yield np.zeros(len(X))

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        W = rng.randn(N, self.ndim) / ell
        return W, sf2

    def get_local(self, x, X1, X2=None):
        ell = np.tile(np.exp(self._logell*-2), self.ndim)
        sf2 = np.exp(self._logsf*2)
        return local_se(ell, sf2, 0.0, x, X1, X2)
