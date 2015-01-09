"""
Simple wrapper class for a Basic GP.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils.models import printable
from ..likelihoods import Gaussian
from ..kernels import SE, Matern
from .exact import ExactGP

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
            raise ValueError('Unknown kernel type')

        super(BasicGP, self).__init__(likelihood, kernel, mu)

    def _params(self):
        # replace the parameters for the base GP model with a simplified
        # structure and rename the likelihood's sigma parameter to sn (ie its
        # the sigma corresponding to the noise).
        params = [('sn', 1, True)]
        params += self._kernel._params()
        params += [('mu', 1, False)]
        return params

    @classmethod
    def from_gp(cls, gp):
        if not isinstance(gp._likelihood, Gaussian):
            raise ValueError('BasicGP instances must have Gaussian likelihood')

        if isinstance(gp._kernel, SE):
            kernel = 'se'
        elif isinstance(gp._kernel, Matern):
            kernel = 'matern%d' % gp._kernel._d
        else:
            raise ValueError('BasicGP instances must have a SE/Matern kernel')

        # get the relevant parameters.
        sn = np.sqrt(gp._likelihood.s2)
        sf = np.exp(gp._kernel._logsf)
        ell = np.exp(gp._kernel._logell)
        mu = gp._mean

        # create the new gp and maybe add data.
        newgp = cls(sn, sf, ell, mu)
        if gp.ndata > 0:
            X, y = gp.data
            newgp.add_data(X, y)
        return newgp
