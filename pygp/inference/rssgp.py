"""
Randomized sparse spectrum GP approximation.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla
import itertools as it

from ._base import GP
from ..likelihoods import Gaussian
from mwhutils.random import rstate


__all__ = ['RSSGP']


class RSSGP(GP):
    """
    GP inference using randomized sparse spectra.
    """
    def __init__(self, likelihood, kernel, mean, nfeatures=100):
        # NOTE: RSSGP only works with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ValueError('exact inference requires a Gaussian likelihood')

        super(RSSGP, self).__init__(likelihood, kernel, mean)
        self._nfeatures = nfeatures

        # sufficient statistics that we'll need.
        self._R = None
        self._m = None
        self._c = None

        # feature parameters
        self._W = None
        self._a = None
        self._b = None

    def reset(self):
        for attr in 'LRbAa':
            setattr(self, '_' + attr, None)
        super(RSSGP, self).reset()

    @classmethod
    def from_gp(cls, gp, nfeatures):
        newgp = cls(gp._likelihood.copy(), gp._kernel.copy(), gp._mean)
        newgp._nfeatures = nfeatures
        if gp.ndata > 0:
            X, y = gp.data
            newgp.add_data(X, y)
        return newgp

    def _get_features(self, X, grad=False):
        """Returns the matrix Phi(X) and its gradient."""
        Z = np.dot(X, self._W.T) + self._b
        Phi = np.cos(Z) * self._a

        if not grad:
            return Phi

        # evaluate the gradient
        dPhi = (-self._a * np.sin(Z))[:, :, None] * self._W[None]

        return Phi, dPhi

    def _update(self):
        rng = rstate()
        sn2 = self._likelihood.s2
        r = self._y - self._mean

        # randomly sample kernel's Fourier feature
        W, alpha = self._kernel.sample_spectrum(self._nfeatures, rng)

        self._W = W
        self._b = rng.rand(self._nfeatures) * 2 * np.pi
        self._a = np.sqrt(2 * alpha / self._nfeatures)

        # evaluate the features
        Phi = self._get_features(self._X)

        # get the components for regression
        A = np.dot(Phi.T, Phi) + sn2 * np.eye(self._nfeatures)

        # cholesky decomposition. note that we only need to compute
        # this once as it is independent from the test point.
        self._R = sla.cholesky(A)
        self._m = sla.cho_solve((self._R, False), Phi.T.dot(r))
        self._c = Phi.dot(self._m)

    def _full_posterior(self, X):
        sn2 = self._likelihood.s2
        mu = np.full(X.shape[0], self._mean)
        Sigma = sn2 * np.eye(X.shape[0])

        if self._X is not None:
            phi = self._get_features(X)
            Rphi = sla.solve_triangular(self._R, phi, trans=True)

            mu += np.dot(self._m.T, phi)
            Sigma += sn2 * np.dot(Rphi.T, Rphi)

        return mu, Sigma

    def _marg_posterior(self, X, grad=False):
        sn2 = self._likelihood.s2
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self._mean)
        s2 = np.full(X.shape[0], sn2)

        if self._X is not None:
            phi = self._get_features(X, grad)
            if grad:
                phi, dphi = phi

            Rphi = sla.solve_triangular(self._R, phi.T, trans=True)

            mu += np.dot(self._m.T, phi.T)
            s2 += sn2 * np.sum(Rphi**2, axis=0)

        if not grad:
            return (mu, s2)

        # Get the prior gradients. Note that this assumes a constant mean and
        # stationary kernel.
        # Note: Not sure this is going to work when X is not a row vector
        # (because of the gradient of Phi(X)).
        dmu = np.zeros_like(X)
        ds2 = np.zeros_like(X)

        if self._X is not None:
            dmu += np.einsum('ijk,j', dphi, self._m)

            dphi = dphi.reshape(self._nfeatures, -1)
            Rdphi = sla.solve_triangular(self._R, dphi, trans=True)
            Rdphi = np.rollaxis(np.reshape(Rdphi, (self._nfeatures,) + X.shape), 2)

            ds2 += 2 * np.sum(Rdphi * Rphi, axis=1).T

        return (mu, s2, dmu, ds2)

    def loglikelihood(self, grad=False):
        # noise hyperparameters
        sn2 = self._likelihood.s2
        ell = np.sqrt(sn2)

        # get the rest of the kernels and the residual.
        r = self._y.copy() - self._mean
        r /= ell

        # the cholesky of Q.
        V = self._get_features(self._X)
        V /= ell

        p = self._nfeatures
        A = self._R
        beta = self._m

        lZ = -np.sum(np.log(np.diag(A))) - self.ndata * np.log(ell)
        lZ -= 0.5 * (np.inner(r, r) - np.inner(beta, beta))
        lZ -= 0.5 * self.ndata * np.log(2*np.pi)

        if not grad:
            return lZ

        raise NotImplementedError
