"""
Implementation of the squared-exponential kernels.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
from abc import abstractmethod

# local imports
from ..utils.models import Parameterized

# exported symbols
__all__ = ['Likelihood', 'RealLikelihood']


class Likelihood(Parameterized):
    """
    Likelihood interface.
    """
    @abstractmethod
    def transform(self, y):
        """
        Perform any cleanup transformations (data arranging, etc) on outputs
        given by the vector `y`.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, f, rng=None):
        """
        Sample noisy outputs given evaluations from the latent function.
        """
        raise NotImplementedError


class RealLikelihood(Likelihood):
    """
    Likelihood model with real-valued outputs.
    """
    def transform(self, y):
        return np.array(y, ndmin=1, dtype=float, copy=False)
