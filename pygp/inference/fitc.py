"""
FITC approximation for sparse pseudo-input GPs.
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
__all__ = ['FITC']


class FITC(GP):
    """
    GP inference using sparse pseudo-inputs.
    """
    def __init__(self, likelihood, kernel, U):
        # NOTE: exact FITC inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ModelError('exact inference requires a Gaussian likelihood')

        super(FITC, self).__init__(likelihood, kernel)
        self._U = np.array(U, ndmin=2, dtype=float, copy=True)
        self._L = None
        self._R = None
        self._b = None

    def _update(self):
        sn2 = self._likelihood.s2
        su2 = sn2 / 1e6

        # kernel wrt the inducing points.
        Kuu = self._kernel.get(self._U)
        p = self._U.shape[0]

        # cholesky for the information gain. note that we only need to compute
        # this once as it is independent from the data.
        self._L = sla.cholesky(Kuu + su2*np.eye(p))

        # evaluate the kernel and residuals at the new points
        Kux = self._kernel.get(self._U, self._X)
        kxx = self._kernel.dget(self._X)
        r = self._y.copy()

        # the cholesky of Q.
        V = sla.solve_triangular(self._L, Kux, trans=True)

        # rescale everything by the diagonal matrix ell.
        ell = np.sqrt(kxx + sn2 - np.sum(V**2, axis=0))
        Kux /= ell
        V /= ell
        r /= ell

        # NOTE: to update things incrementally all we need to do is store these
        # components. A just needs to be initialized at the identity and then
        # we just accumulate here.
        A = np.eye(p) + np.dot(V, V.T)
        a = np.dot(Kux, r)

        # update the posterior.
        self._R = np.dot(sla.cholesky(A), self._L)
        self._b = sla.solve_triangular(self._R, a, trans=True)

    def _posterior(self, X):
        mu = np.zeros(X.shape[0])
        Sigma = self._kernel.get(X)

        if self._X is not None:
            # get the kernel and do two backsolves by the lower-dimensional
            # choleskys that we've stored.
            K = self._kernel.get(self._U, X)
            LK = sla.solve_triangular(self._L, K, trans=True)
            RK = sla.solve_triangular(self._R, K, trans=True)

            # add on the posterior mean contribution and reduce the variance
            # based on the information that we gain from the posterior but add
            # additional uncertainty the further away we are from the inducing
            # points.
            mu += np.dot(RK.T, self._b)
            Sigma += np.dot(RK.T, RK) - np.dot(LK.T, LK)

        return mu, Sigma

    def posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.zeros(X.shape[0])
        s2 = self._kernel.dget(X)

        if self._X is not None:
            # get the kernel and do two backsolves by the lower-dimensional
            # choleskys that we've stored.
            K = self._kernel.get(self._U, X)
            LK = sla.solve_triangular(self._L, K, trans=True)
            RK = sla.solve_triangular(self._R, K, trans=True)

            # add on the posterior mean contribution and reduce the variance
            # based on the information that we gain from the posterior but add
            # additional uncertainty the further away we are from the inducing
            # points.
            mu += np.dot(RK.T, self._b)
            s2 += np.sum(RK**2, axis=0) - np.sum(LK**2, axis=0)

        if not grad:
            return (mu, s2)

        raise NotImplementedError

    def loglikelihood(self, grad=False):
        # noise hyperparameters
        sn2 = self._likelihood.s2
        su2 = sn2 / 1e6

        # number of data points and the inducing points.
        p = self._U.shape[0]

        # cholesky of the pseudo-input kernel.
        Kuu = self._kernel.get(self._U)
        L = sla.cholesky(Kuu + su2*np.eye(p))

        # get the rest of the kernels and the residual.
        Kux = self._kernel.get(self._U, self._X)
        kxx = self._kernel.dget(self._X)
        r = self._y.copy()

        # the cholesky of Q.
        V = sla.solve_triangular(L, Kux, trans=True)

        # rescale everything by the diagonal matrix ell.
        ell = np.sqrt(kxx + sn2 - np.sum(V**2, axis=0))
        Kux /= ell
        V /= ell
        r /= ell

        # Note this A corresponds to chol(self.A) from _update.
        A = sla.cholesky(np.dot(V, V.T) + np.eye(p))
        beta = sla.solve_triangular(A, V.dot(r), trans=True)
        alpha = (r - V.T.dot(sla.solve_triangular(A, beta))) / ell

        lZ = -np.sum(np.log(np.diag(A))) - np.sum(np.log(ell))
        lZ -= 0.5 * (np.inner(r, r) - np.inner(beta, beta))
        lZ -= 0.5 * ell.shape[0] * np.log(2*np.pi)

        if not grad:
            return lZ

        B = sla.solve_triangular(L, V*ell)
        W = sla.solve_triangular(A, V/ell, trans=True)
        w = B.dot(alpha)
        v = 2*su2*np.sum(B**2, axis=0)

        # allocate space for the gradients.
        dlZ = np.empty(1+self._kernel.nhyper)

        # gradient wrt the noise parameter.
        dlZ[0] = (
            - sn2 * (np.sum(1/ell**2) - np.sum(W**2) - np.inner(alpha, alpha))
            - su2 * (np.sum(w**2) + np.sum(B.dot(W.T)**2))
            + 0.5 * (
                np.inner(alpha, v*alpha) + np.inner(np.sum(W**2, axis=0), v)))

        # iterator over gradients of the kernels
        dK = it.izip(
            self._kernel.grad(self._U),
            self._kernel.grad(self._U, self._X),
            self._kernel.dgrad(self._X))

        # gradient wrt the kernel hyperparameters.
        i = 1
        for i, (dKuu, dKux, dkxx) in enumerate(dK, i):
            M = 2*dKux - dKuu.dot(B)
            v = dkxx - np.sum(M*B, axis=0)
            dlZ[i] = (
                np.sum(dkxx/ell**2)
                - np.inner(w, dKuu.dot(w) - 2*dKux.dot(alpha))
                + np.inner(alpha, v*alpha) + np.inner(np.sum(W**2, axis=0), v)
                + np.sum(M.dot(W.T) * B.dot(W.T))) / 2.0

        return lZ, dlZ
