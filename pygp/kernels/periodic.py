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
from ._distances import dist, dist_per_dim
from ..utils.models import Printable

# exported symbols
__all__ = ['Periodic']


class Periodic(RealKernel, Printable):
    def __init__(self, sf, ell, p):
        self._logsf = np.log(float(sf))
        self._logell = np.log(float(ell))
        self._logp = np.log(float(p))
        self.ndim = 1
        self.nhyper = 3

    def _params(self):
        return [
            ('sf',  1),
            ('ell', 1),
            ('p',   1),]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell, self._logp]

    def set_hyper(self, hyper):
        self._logsf = hyper[0]
        self._logell = hyper[1]
        self._logp = hyper[2]

    def get(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        p = np.exp(self._logp)

        K = dist(1, X1, X2) * np.pi / p
        K = sf2 * np.exp(-2*(np.sin(K) / ell)**2)
        return K

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def grad(self, X1, X2=None):
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        p = np.exp(self._logp)

        # get the distance and a few transformations
        D = dist(1, X1, X2) * np.pi / p
        R = np.sin(D) / ell
        S = R**2
        E = 2 * sf2 * np.exp(-2*S)

        yield E
        yield 2*E*S
        yield 2*E*R*D * np.cos(D) / ell

    def dgrad(self, X):
        yield 2 * self.dget(X)
        yield np.zeros(len(X))
        yield np.zeros(len(X))
