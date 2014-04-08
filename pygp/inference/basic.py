"""
Simple wrapper class for a Basic GP.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from .exact import ExactGP
from ..likelihoods import Gaussian
from ..kernels import SEARD
from ..utils.models import Printable

# exported symbols
__all__ = ['BasicGP']


# NOTE: in the definition of the BasicGP class Printable has to come first so
# that we use the __repr__ method defined there and override the base method.

class BasicGP(Printable, ExactGP):
    def __init__(self, sn, ell, sf):
        if np.iterable(ell):
            kernel = SEARD(ell, sf)
        else:
            # FIXME: add the SEIso kernel and use it here if the length scale
            # parameter is not iterable.
            raise NotImplementedError('no support for SEIso kernel yet')

        likelihood = Gaussian(sn)
        super(BasicGP, self).__init__(likelihood, kernel)

    def _params(self):
        return (
            ('sn', np.exp(self._likelihood._logsigma)),
            ('ell', np.exp(self._kernel._logell)),
            ('sf', np.exp(self._kernel._logsf)),)
