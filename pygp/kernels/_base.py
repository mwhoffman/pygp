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
from ..utils.models import Parameterized, dot_params
from ..utils.iters import product, grad_sum, grad_product

# exported symbols
__all__ = ['Kernel', 'RealKernel']


### BASE KERNEL INTERFACE #####################################################

def _collapse(cls, *parts):
    """
    Given a list of kernels return another list of kernels where objects of
    type cls have been "collapsed". This applies to ComboKernel objects which
    represent associative operations.
    """
    collapsed = []
    for part in parts:
        collapsed += part._parts if isinstance(part, cls) else [part]
    return collapsed


class Kernel(Parameterized):
    """
    The base Kernel interface.
    """
    def __add__(self, other):
        return SumKernel(*_collapse(SumKernel, self, other))

    def __mul__(self, other):
        return ProductKernel(*_collapse(ProductKernel, self, other))

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


### COMBINATION KERNELS #######################################################

# FIXME: should ComboKernel objects make a copy of their constituent kernels?
# Otherwise we can do something like kernelA = kernelB + kernelB, but then
# kernelA.set_hyper(...) may have problems due to the fact that kernelA._parts
# contains two references to the same kernel.

# FIXME2: it probably should. but this doesn't neccessarily mean that other
# places should make copies (ie when constructing a GP). at least there it's
# relatively straightforward that the semantics of creating a GP takes a
# reference... but here it seems the semantics of "adding" should return a new
# object.

class ComboKernel(Kernel):
    """
    General implementation of kernels which combine other kernels. Common
    examples are sums and products of kernels.
    """
    def __init__(self, *parts):
        self._parts = [part.copy() for part in parts]
        self.nhyper = sum(p.nhyper for p in self._parts)

        for attr in ['ndim']:
            try:
                setattr(self, attr, getattr(self._parts[0], attr))
            except AttributeError:
                pass

        # FIXME: add some sort of check here so that the kernels can verify
        # whether they can be combined.

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

    def transform(self, X):
        return self._parts[0].transform(X)

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


### OTHER BASE KERNEL TYPES ###################################################

class RealKernel(Kernel):
    """Kernel whose inputs are real-valued vectors."""

    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)
