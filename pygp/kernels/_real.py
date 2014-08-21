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
from ..utils.abc import abstractmethod

# import the generic sum/product kernels and change their names. We'll call the
# real-valued versions SumKernel and ProductKernel as well since they really
# shouldn't be used outside of this module anyway.
from ._combo import SumKernel as SumKernel_
from ._combo import ProductKernel as ProductKernel_
from ._combo import combine
from ._combo import grad_product

# exported symbols
__all__ = ['RealKernel']


class RealKernel(Kernel):
    """Kernel whose inputs are real-valued vectors."""

    def __add__(self, other):
        return SumKernel(*combine(SumKernel, self, other))

    def __mul__(self, other):
        return ProductKernel(*combine(ProductKernel, self, other))

    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)

    @abstractmethod
    def gradx(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to its first argument. Returns
        an (m,n,d)-array.
        """

    @abstractmethod
    def grady(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to its second argument. Returns
        an (m,n,d)-array.
        """

    @abstractmethod
    def gradxy(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to both its first and second
        arguments. Returns an (m,n,d,d)-array. The (a,b,i,j)th element
        corresponds to the derivative with respect to `X1[a,i]` and `X2[b,j]`.
        """

    @abstractmethod
    def sample_spectrum(self, N, rng=None):
        """
        Sample N values from the spectral density of the kernel, returning a
        set of weights W of size (n,d) and a scalar value representing the
        normalizing constant.
        """


def _can_combine(*parts):
    """
    Return whether a set of real-valued kernels can be combined. Here this
    requires them to all be RealKernel objects and have the same number of
    input dimensions.
    """
    return (all(isinstance(_, RealKernel) for _ in parts) and
            all(_.ndim == parts[0].ndim for _ in parts))


class SumKernel(RealKernel, SumKernel_):
    """A sum of real-valued kernels."""

    def __init__(self, *parts):
        if not _can_combine(*parts):
            raise ValueError('cannot add mismatched kernels')

        super(SumKernel, self).__init__(*parts)
        self.ndim = self._parts[0].ndim

    def gradx(self, X1, X2=None):
        return sum(p.gradx(X1, X2) for p in self._parts)

    def grady(self, X1, X2=None):
        return sum(p.grady(X1, X2) for p in self._parts)

    def gradxy(self, X1, X2=None):
        return sum(p.gradxy(X1, X2) for p in self._parts)

    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError


class ProductKernel(RealKernel, ProductKernel_):
    """A product of real-valued kernels."""

    def __init__(self, *parts):
        if not _can_combine(*parts):
            raise ValueError('cannot multiply mismatched kernels')

        super(ProductKernel, self).__init__(*parts)
        self.ndim = self._parts[0].ndim

    def gradx(self, X1, X2=None):
        fiterable = (p.get(X1, X2)[:, :, None] for p in self._parts)
        giterable = ([p.gradx(X1, X2)] for p in self._parts)
        return sum(grad_product(fiterable, giterable))

    def grady(self, X1, X2=None):
        fiterable = (p.get(X1, X2)[:, :, None] for p in self._parts)
        giterable = ([p.grady(X1, X2)] for p in self._parts)
        return sum(grad_product(fiterable, giterable))

    def gradxy(self, X1, X2=None):
        raise NotImplementedError

    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError
