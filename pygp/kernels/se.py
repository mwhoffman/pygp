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
from .__base import RealKernel
from .__distances import sqdist
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


# NOTE: the definitions of the ARD and Iso kernels are basically the same, and
# will remain so until I implement the gradients. Trying to generalize this code
# seems to make the kernel a lot messier to read, so I'll leave it this way
# until I come up with a nicer way to do things.

class SEIso(RealKernel, Printable):
    def __init__(self, ell, sf):
        self._logell = np.log(float(ell))
        self._logsf = np.log(sf)
        self.nhyper = 2

    def _params(self):
        return (
            ('ell', np.exp(self._logell)),
            ('sf', np.exp(self._logsf)),)

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
