"""
Combination classes.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import Kernel
from ..utils.models import dot_params
from ..utils.iters import product, grad_sum, grad_product

# exported symbols
__all__ = ['ComboKernel', 'SumKernel', 'ProductKernel', 'combine']


class ComboKernel(Kernel):
    """
    Implementation of mixin methods for kernels that are themselves
    combinations of other kernels.
    """
    def __init__(self, *parts):
        self._parts = [part.copy() for part in parts]
        self.nhyper = sum(p.nhyper for p in self._parts)

    def __repr__(self):
        string = self.__class__.__name__ + '('
        indent = len(string) * ' '
        substrings = [repr(p) for p in self._parts]
        string += (',\n').join(substrings) + ')'
        string = ('\n'+indent).join(string.splitlines())
        return string

    def _params(self):
        # this is complicated somewhat because I want to return a flat list of
        # parts. so I avoid calling _params() recursively since we could also
        # contain combo objects.
        params = []
        nparts = 0
        parts = list(reversed(self._parts))
        while len(parts) > 0:
            part = parts.pop()
            if isinstance(part, ComboKernel):
                parts.extend(reversed(part._parts))
            else:
                params.extend(dot_params('part%d' % nparts, part._params()))
                nparts += 1
        return params

    def get_hyper(self):
        return np.hstack(p.get_hyper() for p in self._parts)

    def set_hyper(self, hyper):
        a = 0
        for p in self._parts:
            b = a + p.nhyper
            p.set_hyper(hyper[a:b])
            a = b


class SumKernel(ComboKernel):
    """Kernel representing a sum of other kernels."""

    def get(self, X1, X2=None):
        fiterable = (p.get(X1, X2) for p in self._parts)
        return sum(fiterable)

    def dget(self, X):
        fiterable = (p.dget(X) for p in self._parts)
        return sum(fiterable)

    def grad(self, X1, X2=None):
        giterable = (p.grad(X1, X2) for p in self._parts)
        return grad_sum(giterable)

    def dgrad(self, X):
        giterable = (p.dgrad(X) for p in self._parts)
        return grad_sum(giterable)


class ProductKernel(ComboKernel):
    """Kernel representing a product of other kernels."""

    def get(self, X1, X2=None):
        fiterable = (p.get(X1, X2) for p in self._parts)
        return product(fiterable)

    def dget(self, X):
        fiterable = (p.dget(X) for p in self._parts)
        return product(fiterable)

    def grad(self, X1, X2=None):
        fiterable = (p.get(X1, X2) for p in self._parts)
        giterable = (p.grad(X1, X2) for p in self._parts)
        return grad_product(fiterable, giterable)

    def dgrad(self, X):
        fiterable = (p.dget(X) for p in self._parts)
        giterable = (p.dgrad(X) for p in self._parts)
        return grad_product(fiterable, giterable)


def combine(cls, *parts):
    """
    Given a list of kernels return another list of kernels where objects of
    type cls have been "combined". This applies to ComboKernel objects which
    represent associative operations.
    """
    combined = []
    for part in parts:
        combined += part._parts if isinstance(part, cls) else [part]
    return combined
