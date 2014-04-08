"""
Implementation of exact latent-function inference in a Gaussian process model
for regression.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.linalg as sla

# local imports
from .__base import GPModel
from ..likelihoods import Gaussian
from ..utils.exceptions import ModelError

# exported symbols
__all__ = ['ExactGP']


class ExactGP(GPModel):
    def __init__(self, likelihood, kernel):
        # XXX: exact inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ModelError('exact inference requires a Gaussian likelihood')

        super(ExactGP, self).__init__(likelihood, kernel)
        self._R = None
        self._a = None

    def _update(self):
        sn2 = np.exp(self._likelihood._logsigma*2)
        K = self._kernel.get(self._X) + sn2 * np.eye(len(self._X))
        y = self._y
        self._R = sla.cholesky(K)
        self._a = sla.solve_triangular(self._R, y, trans=True)

    def _updateinc(self, X, y):
        sn2 = np.exp(self._likelihood._logsigma*2)
        Kss = self._kernel.get(X) + sn2 * np.eye(len(X))
        Kxs = self._kernel.get(self._X, X)
        y = y
        self._R, self._a = chol_update(self._R, Kxs, Kss, self._a, y)

    def _posterior(self, X, diag=True):
        # grab the prior mean and variance.
        mu = np.zeros(X.shape[0])
        s2 = self._kernel.dget(X) if diag else self._kernel.get(X)

        if self._X is not None:
            K = self._kernel.get(self._X, X)
            V = sla.solve_triangular(self._R, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(V.T, self._a)
            s2 -= np.sum(V**2, axis=0) if diag else np.dot(V.T, V)

        return mu, s2


def chol_update(A, B, C, a, b):
    n = A.shape[0]
    m = C.shape[0]

    B = sla.solve_triangular(A, B, trans=True)
    C = sla.cholesky(C - np.dot(B.T, B))
    c = np.dot(B.T, a)

    # grow the new cholesky and use then use this to grow the vector a.
    A = np.r_[np.c_[A, B], np.c_[np.zeros((m,n)), C]]
    a = np.r_[a, sla.solve_triangular(C, b-c, trans=True)]

    return A, a
