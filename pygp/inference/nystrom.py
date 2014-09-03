"""
Nystrom approximate inference in a Gaussian process model
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
from ..utils.exceptions import ModelError
from ..likelihoods import Gaussian
from ._base import GP

# exported symbols
__all__ = ['NystromGP']


class NystromGP(GP):
    """Nystrom approximation to GP inference."""

    def __init__(self, likelihood, kernel, U):
        # NOTE: exact inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ModelError('exact inference requires a Gaussian likelihood')

        super(NystromGP, self).__init__(likelihood, kernel)
        # save the pseudo-input locations.
        self._U = np.array(U, ndmin=2, dtype=float, copy=True)

        self._Kinv = None
        # approximate eigenvalues and eigenvectors
        self._w = None
        self._V = None

    @property
    def pseudoinputs(self):
        """The pseudo-input points."""
        return self._U

    def _update(self):
        sn2 = self._likelihood.s2
        m = self._U.shape[0]
        eye = np.eye(m)

        Kuu = self._kernel.get(self._U)

        # compute eigenvalues and eigenvectors of smaller problem
        w, V = sla.eigh(Kuu)

        # approximate eigenvalues and eigenvectors of larger problem
        Knm = self._kernel.get(self._X, self._U)
        scalar = self.ndata / m
        self._V = np.dot(Knm, V / w) / np.sqrt(scalar)
        self._w = w * scalar

        # compute the inverse efficiently using Eq. 11 from
        # (Williams and Seeger, 2001)
        b = self._w * self._V
        A = np.dot(b.T, self._V) + sn2 * eye
        self._Kinv = sla.solve(A, b.T)
        self._Kinv = np.eye(self.ndata) - np.dot(self._V, self._Kinv)
        self._Kinv /= sn2

    def _posterior(self, X):
        # grab the prior mean and covariance.
        mu = np.zeros(X.shape[0])
        Sigma = self._kernel.get(X)

        if self._X is not None:
            K = self._kernel.get(self._X, X)
            KinvK = np.dot(self._Kinv, K)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(KinvK.T, self._y)
            Sigma -= np.dot(K.T, KinvK)

        return mu, Sigma

    def posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.zeros(X.shape[0])
        s2 = self._kernel.dget(X)

        if self._X is not None:
            K = self._kernel.get(self._X, X)
            KinvK = np.dot(self._Kinv, K)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(KinvK.T, self._y)
            s2 -= np.sum(K * KinvK, axis=0)

        if not grad:
            return (mu, s2)

        # Get the prior gradients. Note that this assumes a constant mean and
        # stationary kernel.
        dmu = np.zeros_like(X)
        ds2 = np.zeros_like(X)

        if self._X is not None:
            dK = self._kernel.grady(self._X, X)
            dK = dK.reshape(self.ndata, -1)

            KinvdK = np.dot(self._Kinv.T, dK)
            dmu += np.dot(KinvdK.T, self._y).reshape(X.shape)

            dK = np.rollaxis(np.reshape(dK, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(dK * KinvK, axis=1).T

        return (mu, s2, dmu, ds2)

    def loglikelihood(self, grad=False):
        sn2 = self._likelihood.s2

        lZ = -0.5 * np.dot(self._y.T, self._Kinv.dot(self._y))
        lZ -= 0.5 * np.log(2*np.pi) * self.ndata
        lZ -= np.sum(np.log(self._w + sn2))

        # bail early if we don't need the gradient.
        if not grad:
            return lZ

        # # intermediate terms.
        # alpha = sla.solve_triangular(self._R, self._a, trans=False)
        # Q = sla.cho_solve((self._R, False), np.eye(self.ndata))
        # Q -= np.outer(alpha, alpha)
        #
        # dlZ = np.r_[
        #     # derivative wrt the likelihood's noise term.
        #     -self._likelihood.s2 * np.trace(Q),
        #
        #     # derivative wrt each kernel hyperparameter.
        #     [-0.5*np.sum(Q*dK)
        #      for dK in self._kernel.grad(self._X)]]
        #
        # return lZ, dlZ
