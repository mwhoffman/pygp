"""
Definition of the kernel interface.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ..utils.abc import abstractmethod
from ..utils.models import Parameterized

# exported symbols
__all__ = ['Kernel']


### BASE KERNEL INTERFACE #####################################################

class Kernel(Parameterized):
    """
    The base Kernel interface.
    """
    def __call__(self, x1, x2):
        return self.get(x1[None], x2[None])[0]

    @abstractmethod
    def get(self, X1, X2=None):
        """
        Evaluate the kernel.

        Returns the matrix of covariances between points in `X1` and `X2`. If
        `X2` is not given this will return the pairwise covariances between
        points in `X1`.
        """
        pass

    @abstractmethod
    def dget(self, X):
        """Evaluate the self covariances."""
        pass

    @abstractmethod
    def grad(self, X1, X2=None):
        """
        Evaluate the gradient of the kernel.

        Returns an iterator over the gradients of the covariances between
        points in `X1` and `X2`. If `X2` is not given this will iterate over
        the the gradients of the pairwise covariances.
        """
        pass

    @abstractmethod
    def dgrad(self, X):
        """Evaluate the gradients of the self covariances."""
        pass

    @abstractmethod
    def transform(self, X):
        """Format the inputs X as arrays."""
        pass
