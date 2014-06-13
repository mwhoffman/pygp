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
from ._distances import rescale, diff, sqdist, sqdist_foreach

from ..utils.random import rstate
from ..utils.models import Printable

# exported symbols
__all__ = ['Matern']


class Matern(RealKernel, Printable):
    def __init__(self, sf, ell, d=3, ndim=None):
        self._logsf = np.log(float(sf))
        self._logell = np.log(np.ravel(ell))
        self._d = d
        self._iso = False
        self.ndim = self._logell.size
        self.nhyper = 1 + self._logell.size

        if (self._logell.size == 1) and (ndim > 1):
            self._logell = float(self._logell)
            self._iso = True
            self.ndim = ndim

        # FIXME: should I raise an error here if the dimensions are
        # inconsistent?

        # FIXME: check if d is in {1, 3, 5}?

    def _params(self):
        return [
            ('sf',  1),
            ('ell', self.nhyper-1),]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf  = hyper[0]
        self._logell = hyper[1] if self._iso else hyper[1:]

    def _f(self, r):
        return 1                if (self._d == 1) else \
               1+r              if (self._d == 3) else \
               1+r*(1+r/3.)

    def _df(self, r):
        return 1                if (self._d == 1) else \
               r                if (self._d == 3) else \
               r*(1+r)/3.

    def get(self, X1, X2=None):
        X1, X2 = rescale(self._logell - 0.5*np.log(self._d), X1, X2)
        D = np.sqrt(sqdist(X1, X2))
        S = np.exp(self._logsf*2 - D)
        K = S * self._f(D)
        return K

    def grad(self, X1, X2=None):
        X1, X2 = rescale(self._logell - 0.5*np.log(self._d), X1, X2)
        D = np.sqrt(sqdist(X1, X2))
        S = np.exp(self._logsf*2 - D)
        K = S * self._f(D)
        M = S * self._df(D)

        yield 2*K                                  # derivative wrt logsf
        if self._iso:
            yield M*D                              # derivative wrt logell (iso)
        else:
            for D_ in sqdist_foreach(X1, X2):
                yield np.where(D<1e-12, 0, M*D_/D) # derivative wrt logell (ard)

    def gradx(self, X1, X2=None):
        X1, X2 = rescale(self._logell - 0.5*np.log(self._d), X1, X2)
        D1 = diff(X1, X2)
        ell = np.exp(self._logell)

        D = np.sqrt(np.sum(D1**2, axis=-1))
        S = np.exp(self._logsf*2 - D)
        M = np.where(D<1e-12, 0, S * self._df(D) / D)
        G = M[:,:,None] * D1 / ell * np.sqrt(self._d)

        return G

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def dgrad(self, X1):
        yield 2 * self.dget(X1)
        for i in xrange(self.nhyper-1):
            yield np.zeros(len(X1))

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        a = self._d / 2.
        g = np.tile(rng.gamma(a, 1/a, N), (self.ndim, 1)).T
        W = (rng.randn(N, self.ndim) / ell) / np.sqrt(g)
        return W, sf2
