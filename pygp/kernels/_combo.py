"""
Combination classes.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import itertools as it
import functools as ft
import operator as op

# local imports
from ._base import Kernel

# exported symbols
__all__ = ['ComboKernel', 'SumKernel', 'ProductKernel', 'combine']


### HELPER METHODS ############################################################

def product(fiterable):
    """
    The equivalent object to sum but for products.
    """
    return ft.reduce(op.mul, fiterable, 1)


def product_but(fiterable):
    """
    Given an iterator over function evaluations return an array such that
    `M[i]` is the product of every evaluation except for the ith one.
    """
    A = list(fiterable)

    # allocate memory for M and fill everything but the last element with
    # the product of A[i+1:]. Note that we're using the cumprod in place.
    M = np.empty_like(A)
    np.cumprod(A[:0:-1], axis=0, out=M[:-1][::-1])

    # use an explicit loop to iteratively set M[-1] equal to the product of
    # A[:-1]. While doing this we can multiply M[i] by A[:i].
    M[-1] = A[0]
    for i in xrange(1, len(A)-1):
        M[i] *= M[-1]
        M[-1] *= A[i]

    return M


### GENERAL COMBINATION KERNEL ################################################

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
                params.extend([("part%d.%s" % (nparts, p[0]),) + p[1:]
                               for p in part._params()])
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


### SUM AND PRODUCT KERNELS ###################################################

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
        return it.chain.from_iterable(giterable)

    def dgrad(self, X):
        giterable = (p.dgrad(X) for p in self._parts)
        return it.chain.from_iterable(giterable)


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
        for Mi, grads in zip(product_but(fiterable), giterable):
            for dM in grads:
                yield Mi*dM

    def dgrad(self, X):
        fiterable = (p.dget(X) for p in self._parts)
        giterable = (p.dgrad(X) for p in self._parts)
        for Mi, grads in zip(product_but(fiterable), giterable):
            for dM in grads:
                yield Mi*dM


### HELPER FOR ASSOCIATIVE OPERATIONS #########################################

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
