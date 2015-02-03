"""
Implementation of the hinge-like prior means.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
from scipy.spatial.distance import cdist

# local imports
from ._base import Mean
from ..kernels._distances import rescale, sqdist
from ..utils.models import printable

# exported symbols
__all__ = ['HingeQuadraticIso']


@printable
class HingeQuadraticIso(Mean):
    def __init__(self, bias, centre, inner, shell, ndim=None):
        self._bias = bias
        self._centre = np.array(centre, ndmin=1)
        self._inner = inner
        self._shell = shell
        self.ndim = np.size(self._centre)

        # the following should really only be used when centering around
        # the origin.
        if (np.size(centre) == 1) and (ndim is not None):
            self._centre = np.full(ndim, np.float(centre))
            self.ndim = ndim

        self.nhyper = (3 + self.ndim)

    def _params(self):
        return [
            ('bias', 1, False),
            ('centre', self.ndim, False),
            ('inner', 1, False),
            ('shell', 1, False)]

    def get_hyper(self):
        return np.r_[
            self._bias,
            self._centre,
            self._inner,
            self._shell]

    def set_hyper(self, hyper):
        self._bias = hyper[0]
        self._centre = hyper[1:self.ndim+1]
        self._inner = hyper[-2]
        self._shell = hyper[-1]

    def get(self, X):
        X0 = np.array(self._centre, ndmin=2)
        D = cdist(X, X0, 'euclidean').ravel()
        idx = (D > self._inner)
        mean = np.full(X.shape[0], self._bias)
        mean[idx] -= ((D[idx] - self._inner) / (self._shell * self._inner)) ** 2
        return mean

    def grad(self, X):
        """Gradient wrt the value of the constant mean."""
        yield np.ones((1, X.shape[0]))                  # derivative wrt bias

        X0 = np.array(self._centre, ndmin=2)
        D = cdist(X, X0, 'euclidean').ravel()
        idx = (D > self._inner)
        dr = self._shell * self._inner
        dist = D[idx] - self._inner

        grad = np.zeros_like(X.T)
        chain1 = 2 * dist / dr ** 2
        direction = (X[idx] - X0).T / D[idx]
        grad[:, idx] += chain1 * direction
        yield grad                                      # derivative wrt centre

        grad = np.zeros(X.shape[0])
        dist2 = dist**2
        chain2 = 2 * dist2 / self._shell**2 / self._inner**3
        grad[idx] += chain1
        grad[idx] += chain2
        yield grad                                      # derivative wrt inner

        grad = np.zeros(X.shape[0])
        chain3 = 2 * dist2 / self._shell**3 / self._inner**2
        grad[idx] += chain3
        yield grad                                      # derivative wrt shell

    def gradx(self, X):
        """Gradient wrt the inputs X."""
        X0 = np.array(self._centre, ndmin=2)
        D = cdist(X, X0, 'euclidean').ravel()
        idx = (D > self._inner)

        grad = np.zeros_like(X)
        chain1 = 2 * (D[idx] - self._inner) / (self._shell * self._inner)**2
        direction = (X[idx] - X0).T / D[idx]
        grad[idx] -= np.transpose(chain1 * direction)
        return grad
