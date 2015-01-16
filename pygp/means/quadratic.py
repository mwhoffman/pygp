"""
Implementation of the quadratic mean.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import Mean
from ..kernels._distances import rescale, sqdist
from ..utils.models import printable

# exported symbols
__all__ = ['Quadratic']


@printable
class Quadratic(Mean):
    def __init__(self, bias, centre, widths, ndim=None):
        self._bias = bias
        self._centre = np.array(centre, ndmin=1)
        self._widths = np.array(widths, ndmin=1)
        self._iso = False
        self.ndim = np.size(self._widths)

        if ndim is not None:
            if np.size(widths) == 1:
                self._widths = float(self._widths)
                self._iso = True
                self.ndim = ndim
            else:
                raise ValueError('ndim only usable with scalar widths')

            # the following should really only be used when centering around
            # the origin.
            if np.size(centre) == 1:
                self._centre = np.full(ndim, np.float(centre))

        self.nhyper = 1 + self.ndim + np.size(self._widths)
        assert np.size(self._centre) == self.ndim, \
            'dimension of `centre` should match that of `widths` or `ndim`'

    def _params(self):
        return [
            ('bias', 1, False),
            ('centre', self.ndim, False),
            ('widths', 1 if self._iso else self.ndim, False)]

    def get_hyper(self):
        return np.r_[
            self._bias,
            self._centre,
            self._widths]

    def set_hyper(self, hyper):
        self._bias = hyper[0]
        self._centre = hyper[1:self.ndim+1]
        self._widths = hyper[-1] if self._iso else hyper[-self.ndim:]

    def get(self, X):
        X0 = np.array(self._centre, ndmin=2)
        X, X0 = rescale(self._widths, X, X0)
        return self._bias - sqdist(X, X0)

    def grad(self, X):
        """Gradient wrt the value of the constant mean."""
        yield np.ones((1, X.shape[0]))                  # derivative wrt bias

        X0 = np.array(self._centre, ndmin=2)
        D = (X - X0) / (self._widths ** 2)
        yield 2 * D.T                                   # derivative wrt centre

        X, X0 = rescale(self._widths, X, X0)
        D2 = (X - X0) ** 2
        K = 2 / self._widths
        if self._iso:
            yield K * np.sum(D2, axis=1).T              # derivative wrt widths
        else:
            G = K * D2
            yield G.T

    def gradx(self, X):
        """Gradient wrt the inputs X."""
        X0 = np.array(self._centre, ndmin=2)
        D = X - X0
        return -2 * D / (self._widths ** 2)
