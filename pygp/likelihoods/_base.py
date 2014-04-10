"""
Implementation of the squared-exponential kernels.
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
__all__ = ['Likelihood', 'RealLikelihood']


class Likelihood(Parameterized):
    """
    Likelihood interface.
    """
    @abc.abstractmethod
    def transform(self, y):
        pass


class RealLikelihood(Likelihood):
    def transform(self, y):
        return np.array(y, ndmin=1, dtype=float, copy=False)
