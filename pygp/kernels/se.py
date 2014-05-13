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
from ..utils.random import rstate
from ..utils.models import Printable

# exported symbols
__all__ = ['SEARD', 'SEIso']


class SEARD(RealKernel, Printable):
    def __init__(self, sf, ell):
        self._logsf = np.log(sf)
        self._logell = np.log(np.ravel(ell))
        self.ndim = len(self._logell)
        self.nhyper = self.ndim + 1

    def _params(self):
        return [
            ('sf',  'log', 1),
            ('ell', 'log', self.ndim),]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf  = hyper[0]
        self._logell = hyper[1:]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        return sf2 * np.exp(-0.5*sqdist(ell, X1, X2))

    def dget(self, X1):
        sf2 = np.exp(self._logsf*2)
        return sf2 * np.ones(len(X1))

    def grad(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        for D in sqdist_per_dim(ell, X1, X2):
            D *= sf2 * np.exp(-0.5*D)
            yield D
        yield 2*self.get(X1, X2)

    def dgrad(self, X):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        for i in xrange(self.ndim):
            yield np.zeros(len(X))
        yield 2 * sf2 * np.ones(len(X))

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        W = rng.randn(N, self.ndim) / ell
        return W, sf2


# NOTE: the definitions of the ARD and Iso kernels are basically the same, and
# will remain so until I implement the gradients. Trying to generalize this code
# seems to make the kernel a lot messier to read, so I'll leave it this way
# until I come up with a nicer way to do things.

class SEIso(RealKernel, Printable):
    def __init__(self, sf, ell, ndim=1):
        self._logsf = np.log(sf)
        self._logell = np.log(float(ell))
        self.ndim = ndim
        self.nhyper = 2

    def _params(self):
        return (
            ('sf',  'log', 1),
            ('ell', 'log', 1),
            )

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf  = hyper[0]
        self._logell = hyper[1]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        return sf2 * np.exp(-0.5*sqdist(ell, X1, X2))

    def dget(self, X1):
        sf2 = np.exp(self._logsf*2)
        return sf2 * np.ones(len(X1))

    def grad(self, X1, X2=None):
        ell = np.exp(self._logell)
        sf2 = np.exp(self._logsf*2)
        D = sqdist(ell, X1, X2)
        K = sf2 * np.exp(-0.5*D)
        yield 2*K
        yield D*K

    def dgrad(self, X):
        yield 2*self.dget(X)
        yield np.zeros(len(x1))

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        W = rng.randn(N, self._ndim) / ell
        return W, sf2
