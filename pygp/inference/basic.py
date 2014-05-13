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
from ..kernels import SEARD, SEIso
from ..utils.models import Printable

# exported symbols
__all__ = ['BasicGP']


# NOTE: in the definition of the BasicGP class Printable has to come first so
# that we use the __repr__ method defined there and override the base method.

class BasicGP(Printable, ExactGP):
    def __init__(self, sn, sf, ell, ndim=None):
        likelihood = Gaussian(sn)

        kernel = SEARD(sf, ell) if np.iterable(ell) else \
                 SEIso(sf, ell, 1 if (ndim is None) else ndim)

        super(BasicGP, self).__init__(likelihood, kernel)

    def _params(self):
        # replace the parameters for the base GP model with a simplified
        # structure and rename the likelihood's sigma parameter to sn (ie its
        # the sigma corresponding to the noise).
        params = [('sn', 'log', 1)]
        params.extend(self._kernel._params())
        return params
