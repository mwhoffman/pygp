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
from .__base import Kernel, sqdist
from ..utils.models import Printable

# exported symbols
__all__ = ['SEARD']


class SEARD(Kernel, Printable):
    def __init__(self, ell, sf):
        self._logell = np.log(np.ravel(ell))
        self._logsf = np.log(sf)

    def _params(self):
        return (
            ('ell', np.exp(self._logell)),
            ('sf', np.exp(self._logsf)))

    def get(self, x1, x2=None):
        D = sqdist(np.exp(self._logell), x1, x2)
        return np.exp(self._logsf*2 - 0.5*D)

    def dget(self, x1):
        return np.exp(self._logsf*2) * np.ones(len(x1))
