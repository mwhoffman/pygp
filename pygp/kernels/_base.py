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
from ..utils.iters import product, grad_sum, grad_product

# exported symbols
__all__ = ['Kernel', 'RealKernel']


#--BASE KERNEL INTERFACE--------------------------------------------------------

def _collapse(Combiner, *parts):
    collapsed = []
    for part in parts:
        collapsed += part._parts if isinstance(part, Combiner) else [part]
    return collapsed


class Kernel(Parameterized):
    """
    Kernel interface.
    """
    def __add__(self, other):
        return SumKernel(*_collapse(SumKernel, self, other))

    def __mul__(self, other):
        return ProductKernel(*_collapse(ProductKernel, self, other))

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


#--COMBINATION KERNELS----------------------------------------------------------

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
    def __init__(self, *parts):
        self._parts = parts
        self.nhyper = sum(p.nhyper for p in self._parts)

        # FIXME: add some sort of check here so that the kernels can verify
        # whether they can be combined.

    def __repr__(self):
        string = self.__class__.__name__ + '('
        indent = len(string) * ' '
        substrings = [repr(p) for p in self._parts]
        string += (',\n').join(substrings) + ')'
        string = ('\n'+indent).join(string.splitlines())
        return string

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


#--OTHER BASE KERNEL TYPES------------------------------------------------------

class RealKernel(Kernel):
    def transform(self, X):
        return np.array(X, ndmin=2, dtype=float, copy=False)
