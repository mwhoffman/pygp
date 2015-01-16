"""
Implementation of the constant mean.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import Mean
from ..utils.models import printable

# exported symbols
__all__ = ['Constant']


@printable
class Constant(Mean):
    def __init__(self, bias):
        self._bias = bias
        self.nhyper = 1

    def _params(self):
        return [('bias', 1, False)]

    def get_hyper(self):
        return np.r_[self._bias]

    def set_hyper(self, hyper):
        self._bias = hyper

    def get(self, X):
        return np.full(X.shape[0], self._bias)

    def grad(self, X):
        """Gradient wrt the value of the constant mean."""
        yield np.ones(X.shape[0])

    def gradx(self, X):
        """Gradient wrt the inputs X."""
        return np.zeros_like(X)
