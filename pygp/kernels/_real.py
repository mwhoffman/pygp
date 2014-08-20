"""
Base class for real-valued kernels.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import Kernel
from ._combo import SumKernel, ProductKernel, combine
from ..utils.abc import abstractmethod

# exported symbols
__all__ = ['RealKernel']


class RealKernel(Kernel):
    """Kernel whose inputs are real-valued vectors."""

    def __add__(self, other):
        return RealSumKernel(*combine(RealSumKernel, self, other))

    def __mul__(self, other):
        return RealProductKernel(*combine(RealProductKernel, self, other))

    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)

    @abstractmethod
    def gradx(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to its first argument. This
        corresponds to the covariance between the function gradient at X1 and
        the function evaluated at X2. Returns an (m,n,d)-array.
        """
        pass

    @abstractmethod
    def gradxy(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to both its first and second
        arguments. This corresponds to the covariance between gradient values
        evaluated at X1 and at X2. Returns an (m,n,d,d)-array.
        """
        pass

    @abstractmethod
    def sample_spectrum(self, N, rng=None):
        """
        Sample N values from the spectral density of the kernel, returning a
        set of weights W of size (n,d) and a scalar value representing the
        normalizing constant.
        """
        pass


class RealSumKernel(RealKernel, SumKernel):
    def gradx(self, X1, X2=None):
        raise NotImplementedError

    def gradxy(self, X1, X2=None):
        raise NotImplementedError

    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError


class RealProductKernel(RealKernel, ProductKernel):
    def gradx(self, X1, X2=None):
        raise NotImplementedError

    def gradxy(self, X1, X2=None):
        raise NotImplementedError

    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError
