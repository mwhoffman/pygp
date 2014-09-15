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
import itertools as it

# local imports
from ..utils.exceptions import ModelError
from ..likelihoods import Gaussian
from ._base import GP

# exported symbols
__all__ = ['DTC']


class DTC(GP):
    """Deterministic training conditional approximation to GP inference."""

    def __init__(self, likelihood, kernel, mean, U):
        # NOTE: exact inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ModelError('exact inference requires a Gaussian likelihood')

        super(DTC, self).__init__(likelihood, kernel, mean)
        # save the pseudo-input locations.
        self._U = np.array(U, ndmin=2, dtype=float, copy=True)

        self._Ruu = None
        self._Rux = None
        self._a = None

    @property
    def pseudoinputs(self):
        """The pseudo-input points."""
        return self._U

    def _update(self):
        p = self._U.shape[0]
        su2 = self._likelihood.s2 * 1e-6

        # choleskies of Kuu and (Kuu + Kfu * Kuf / sn2), respectively,
        # see Eq 20b of (Quinonero-Candela and Rasmussen, 2005)
        Kuu = self._kernel.get(self._U)
        self._Ruu = sla.cholesky(Kuu + su2 * np.eye(p))

        # formulate data-dependent problem
        Kux = self._kernel.get(self._U, self._X)
        S = Kuu + np.dot(Kux, Kux.T) / self._likelihood.s2

        # compute cholesky of data dependent problem
        r = self._y - self._mean
        self._Rux = sla.cholesky(S + su2 * np.eye(p))
        self._a = sla.solve_triangular(self._Rux,
                                       np.dot(Kux, r),
                                       trans=True)

    def _full_posterior(self, X):
        # grab the prior mean and covariance.
        mu = np.full(X.shape[0], self._mean)
        Sigma = self._kernel.get(X)

        if self._X is not None:
            K = self._kernel.get(self._U, X)
            b = sla.solve_triangular(self._Ruu, K, trans=True)
            c = sla.solve_triangular(self._Rux, K, trans=True)

            # add the contribution to the mean coming from the posterior and
            # subtract off the information gained in the posterior from the
            # prior variance.
            mu += np.dot(c.T, self._a) / self._likelihood.s2
            Sigma += -np.dot(b.T, b) + np.dot(c.T, c)

        return mu, Sigma

    def _marg_posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self._mean)
        s2 = self._kernel.dget(X)

        if self._X is not None:
            K = self._kernel.get(self._U, X)
            b = sla.solve_triangular(self._Ruu, K, trans=True)
            c = sla.solve_triangular(self._Rux, K, trans=True)

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

            dc = sla.solve_triangular(self._Rux, dK, trans=True)
            dmu += np.dot(dc.T, self._a).reshape(X.shape)

            dc = np.rollaxis(np.reshape(dc, (-1,) + X.shape), 2)
            ds2 += -2 * np.sum(db * b, axis=1).T + \
                    2 * np.sum(dc * c, axis=1).T

        return (mu, s2, dmu, ds2)

    def loglikelihood(self, grad=False):
        # noise hyperparameters
        sn2 = self._likelihood.s2
        su2 = sn2 * 1e-6
        ell = np.sqrt(sn2)

        # get the rest of the kernels and the residual.
        Kux = self._kernel.get(self._U, self._X)
        r = self._y.copy() - self._mean
        r /= ell

        # the cholesky of Q.
        V = sla.solve_triangular(self._Ruu, Kux, trans=True)
        V /= ell

        p = self._U.shape[0]
        A = sla.cholesky(np.eye(p) + np.dot(V, V.T))
        beta = sla.solve_triangular(A, V.dot(r), trans=True)

        lZ = -np.sum(np.log(np.diag(A))) - self.ndata * np.log(ell)
        lZ -= 0.5 * (np.inner(r, r) - np.inner(beta, beta))
        lZ -= 0.5 * self.ndata * np.log(2*np.pi)

        if not grad:
            return lZ

        alpha = (r - V.T.dot(sla.solve_triangular(A, beta)))
        B = sla.solve_triangular(self._Ruu, V)
        W = sla.solve_triangular(A, V, trans=True)
        VW = np.dot(V, W.T)
        BW = np.dot(B, W.T)
        w = B.dot(alpha)
        v = V.dot(alpha)

        # allocate space for the gradients.
        dlZ = np.zeros(self.nhyper)

        # gradient wrt the noise parameter.
        dlZ[0] = -(
            # gradient of the mahalanobis term
            - np.inner(r, r)
            + np.inner(beta, beta)
            + np.inner(v, v)
            + su2 * np.inner(w, w)
            # gradient of the log determinant term
            + self.ndata
            - np.sum(V**2)
            + np.sum(VW**2)
            - su2 * (np.sum(B**2) - np.sum(BW**2)))

        # iterator over gradients of the kernels
        dK = it.izip(
            self._kernel.grad(self._U),
            self._kernel.grad(self._U, self._X))

        # gradient wrt the kernel hyperparameters.
        i = 1
        for i, (dKuu, dKux) in enumerate(dK, i):
            M = 2 * dKux / ell - dKuu.dot(B)
            dlZ[i] = -0.5 * (
                - np.inner(w, np.dot(M, alpha))
                + np.sum(M*B)
                - np.sum(M.dot(W.T) * B.dot(W.T)))

        # gradient wrt the constant mean.
        dlZ[-1] = np.sum(alpha) / ell

        return lZ, dlZ
