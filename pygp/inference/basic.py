"""
Simple wrapper class for a Basic GP.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# local imports
from ..utils.models import printable
from ..likelihoods import Gaussian
from ..kernels import SE, Matern
from .exact import ExactGP

# exported symbols
__all__ = ['BasicGP']


@printable
class BasicGP(ExactGP):
    """
    Basic GP frontend which assumes an ARD kernel and a Gaussian likelihood
    (and hence performs exact inference).
    """
    def __init__(self, sn, sf, ell, mu=0, ndim=None, kernel='se'):
        likelihood = Gaussian(sn)
        kernel = (
            SE(sf, ell, ndim) if (kernel == 'se') else
            Matern(sf, ell, 1, ndim) if (kernel == 'matern1') else
            Matern(sf, ell, 3, ndim) if (kernel == 'matern3') else
            Matern(sf, ell, 5, ndim) if (kernel == 'matern5') else None)

        if kernel is None:
            raise RuntimeError('Unknown kernel type')

        super(BasicGP, self).__init__(likelihood, kernel, mu)

    def _params(self):
        # replace the parameters for the base GP model with a simplified
        # structure and rename the likelihood's sigma parameter to sn (ie its
        # the sigma corresponding to the noise).
        params = [('sn', 1, True)]
        params += self._kernel._params()
        params += [('mu', 1, False)]
        return params
