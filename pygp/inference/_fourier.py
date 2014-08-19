"""
Approximations to the GP using random Fourier features.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.linalg as sla

# local imports
from ..utils.random import rstate
from ..utils.exceptions import ModelError
from ..likelihoods import Gaussian

# exported symbols
__all__ = ['FourierSample']


class FourierSample(object):
    def __init__(self, N, likelihood, kernel, X, y, rng=None):
        # if given a seed or an instantiated RandomState make sure that we use
        # it here, but also within the sample_spectrum code.
        rng = rstate(rng)

        if not isinstance(likelihood, Gaussian):
            # FIXME: generalize this?
            raise ModelError('Fourier samples only defined for Gaussian'
                             'likelihoods')

        # this randomizes the feature.
        self.W, self.alpha = kernel.sample_spectrum(N, rng)
        self.b = rng.rand(N) * 2 * np.pi

        if X is not None:
            # FIXME: this bit of code needs to be changed/replaced if we have
            # non-Gaussian likelihoods.
            sigma = np.exp(likelihood._logsigma)

            Phi = self.phi(X)
            A = np.dot(Phi.T, Phi)
            A += sigma**2 * np.eye(Phi.shape[1])
            R = sla.cholesky(A)

            # FIXME: we can do a smarter update here when the number of points
            # is less than the number of features.

            self.theta = sla.cho_solve((R, False), np.dot(Phi.T, y))
            self.theta += sla.solve_triangular(R, sigma*rng.randn(N))

        else:
            self.theta = rng.randn(N)

    def phi(self, X):
        # x is n-by-D,
        # W is N-by-D,
        # Phi, the return value, should be n-by-N.
        rnd = np.dot(X, self.W.T) + self.b
        Phi = np.cos(rnd) * np.sqrt(2 * self.alpha / self.W.shape[0])
        return Phi

    def get(self, X):
        Phi = self.phi(np.array(X, ndmin=2, copy=False))
        return np.dot(Phi, self.theta)

    def __call__(self, x):
        return self.get(x)[0]
