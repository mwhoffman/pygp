"""
Base class for string  kernels.
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

# import the generic sum/product kernels
from ._combo import SumKernel as SumKernel_
from ._combo import ProductKernel as ProductKernel_
from ._combo import combine

# exported symbols
__all__ = ['StringKernel']

# Make a wrapper class for numpy arrays, such that the translate
# method kan store (cache) extra information in the array
class CachedArray(np.ndarray):
    def __new__(cls, input_array, cache={}):
        obj = np.asarray(input_array).view(cls)
        obj.cache = cache
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.cache = getattr(obj, 'cache', None)


class StringKernel(Kernel):
    """Kernel whose inputs are strings."""

    def __add__(self, other):
        return SumKernel(*combine(SumKernel, self, other))

    def __mul__(self, other):
        return ProductKernel(*combine(ProductKernel, self, other))

    def transform(self, X):
        if type(X) == CachedArray:
            return X
        else:
            return CachedArray(X)


def _can_combine(*parts):
    """
    Return whether a set of string kernels can be combined. Here this
    requires them to all be StringKernel objects.
    """
    return all(isinstance(_, StringKernel) for _ in parts)


class SumKernel(StringKernel, SumKernel_):
    """A sum of string kernels."""

    def __init__(self, *parts):
        if not _can_combine(*parts):
            raise ValueError('cannot add mismatched kernels')

        super(SumKernel, self).__init__(*parts)

    def transform(self, X):
        X = super(SumKernel, self).transform(X)        
        for p in self._parts:
            X = p.transform(X)
        return X


class ProductKernel(StringKernel, ProductKernel_):
    """A product of string kernels."""

    def __init__(self, *parts):
        if not _can_combine(*parts):
            raise ValueError('cannot multiply mismatched kernels')

        super(ProductKernel, self).__init__(*parts)

    def transform(self, X):
        X = super(ProductKernel, self).transform(X)        
        for p in self._parts:
            X = p.transform(X)
        return X
