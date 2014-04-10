"""
Definition of the kernel interface.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import abc

# local imports
from ..utils.models import Parameterized

# exported symbols
__all__ = ['Kernel', 'RealKernel']


class Kernel(Parameterized):
    """
    Kernel interface.
    """
    @abc.abstractmethod
    def get(self, X1, X2=None):
        pass

    @abc.abstractmethod
    def dget(self, X):
        pass

    @abc.abstractmethod
    def grad(self, X1, X2=None):
        pass

    @abc.abstractmethod
    def dgrad(self, X):
        pass

    @abc.abstractmethod
    def transform(self, X):
        pass


class RealKernel(Kernel):
    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)
