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

# local imports
from ._base import GP

# exported symbols
__all__ = ['FITC']


class FITC(GP):
    """
    GP inference using sparse pseudo-inputs.
    """
    def __init__(self, likelihood, kernel, U):
        super(FITC, self).__init__(likelihood, kernel)
        self._U = np.array(U, ndmin=2, dtype=float, copy=True)
        self._L = None
        self._R = None
        self._b = None

    def _update(self):
        sn2 = np.exp(self._likelihood._logsigma*2)
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
        r = self._y

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
        raise NotImplementedError
