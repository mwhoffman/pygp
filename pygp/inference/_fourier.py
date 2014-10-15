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
    def __init__(self, N, likelihood, kernel, mean, X, y, rng=None):
        # if given a seed or an instantiated RandomState make sure that we use
        # it here, but also within the sample_spectrum code.
        rng = rstate(rng)

        if not isinstance(likelihood, Gaussian):
            raise ModelError('Fourier samples only defined for Gaussian'
                             'likelihoods')

        # this randomizes the feature.
        W, alpha = kernel.sample_spectrum(N, rng)

        self._W = W
        self._b = rng.rand(N) * 2 * np.pi
        self._a = np.sqrt(2 * alpha / N)
        self._mean = mean
        self._theta = None

        if X is not None:
            # evaluate the features
            Z = np.dot(X, self._W.T) + self._b
            Phi = np.cos(Z) * self._a

            # get the components for regression
            A = np.dot(Phi.T, Phi) + likelihood.s2 * np.eye(Phi.shape[1])
            R = sla.cholesky(A)
            r = y - mean
            p = np.sqrt(likelihood.s2) * rng.randn(N)

            # FIXME: we can do a smarter update here when the number of points
            # is less than the number of features.

            self._theta = sla.cho_solve((R, False), np.dot(Phi.T, r))
            self._theta += sla.solve_triangular(R, p)

        else:
            self._theta = rng.randn(N)

    def get(self, X):
        """
        Evaluate the function at a collection of points.
        """
        X = np.array(X, ndmin=2, copy=False)
        Z = np.dot(X, self._W.T) + self._b
        Phi = np.cos(Z) * self._a
        F = np.dot(Phi, self._theta) + self._mean

        return F

    def __call__(self, x):
        return self.get(x)[0]
