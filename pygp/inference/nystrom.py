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
        # NOTE -- Bobak: if self._U is changed at any point, self._Ruu must be
        # recomputed as well.
        Kuu = self._kernel.get(self._U)


        # choleskies of Kuu and (Kuu + Kfu * Kuf / sn2), respectively,
        # see Eq 20b of (Quinonero-Candela and Rasmussen, 2005)
        self._Ruu = sla.cholesky(Kuu)
        self._Ruf = None
        self._a = None

    @property
    def pseudoinputs(self):
        """The pseudo-input points."""
        return self._U

    def _update(self):
        # compute cholesky of data dependent problem
        Kuf = self._kernel.get(self._U, self._X)
        S = Kuu + np.dot(Kuf, Kuf.T) / self._likelihood.s2
        self._Ruf = sla.cholesky(S)
        self._a = sla.solve_triangular(self._Ruf,
                                       np.dot(Kuf, self._y),
                                       trans=True)

    def _posterior(self, X):
        # grab the prior mean and covariance.
        mu = np.zeros(X.shape[0])
        Sigma = self._kernel.get(X)

        if self._X is not None:
            K = self._kernel.get(self._U, X)
            b = sla.solve_triangular(self._Ruu, K, trans=True)
            c = sla.solve_triangular(self._Ruf, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(c.T, self._a)
            Sigma += -np.dot(b.T, b) + np.dot(c.T, c) * self._likelihood.s2

        return mu, Sigma

    def posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.zeros(X.shape[0])
        s2 = self._kernel.dget(X)

        if self._X is not None:
            K = self._kernel.get(self._U, X)
            b = sla.solve_triangular(self._Ruu, K, trans=True)
            c = sla.solve_triangular(self._Ruf, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(c.T, self._a) / self._likelihood.s2
            s2 += -np.sum(b * b, axis=0) + np.sum(c * c, axis=0)

        if not grad:
            return (mu, s2)

        # Get the prior gradients. Note that this assumes a constant mean and
        # stationary kernel.
        dmu = np.zeros_like(X)
        ds2 = np.zeros_like(X)

        if self._X is not None:
            dK = self._kernel.grady(self._U, X)
            dK = dK.reshape(self._U.shape[0], -1)

            db = sla.solve_triangular(self._Ruu, dK, trans=True)
            db = np.rollaxis(np.reshape(db, (-1,) + X.shape), 2)

            dc = sla.solve_triangular(self._Ruf, dK, trans=True)
            dmu += np.dot(dc.T, self._a).reshape(X.shape)

            dc = np.rollaxis(np.reshape(dc, (-1,) + X.shape), 2)
            ds2 += -2 * np.sum(db * b, axis=1).T + \
                    2 * np.sum(dc * c, axis=1).T

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
