"""
Definition of the mean interface.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
from abc import abstractmethod

# local imports
from ..utils.models import Parameterized

# exported symbols
__all__ = ['Mean']


class Mean(Parameterized):
    """
    The base Mean interface.
    """
    def __call__(self, x):
        """Returns the mean evaluated at a single point `x`."""
        return self.get(x[None])[0]

    @abstractmethod
    def get(self, X):
        """Returns the mean evaluated at points in `X`."""
        raise NotImplementedError

    @abstractmethod
    def grad(self, X):
        """Returns the gradient of the mean evaluated at points in `X`."""
        raise NotImplementedError
