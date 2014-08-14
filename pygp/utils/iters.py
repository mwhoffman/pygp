"""
Tools for creating/reducing iterator objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

import itertools as it
import functools as ft
import operator

# exported symbols
__all__ = ['product', 'grad_sum', 'grad_product']


def product(fiterable):
    """
    The equivalent object to sum but for products.
    """
    return ft.reduce(operator.mul, fiterable, 1)


def grad_sum(giterable):
    # this is not really necessary since the gradient of sums is so trivial,
    # but this makes the code epsilon more readable since it looks the same as
    # the product gradients.
    return it.chain.from_iterable(giterable)


def grad_product(fiterable, giterable):
    # This implements a helper for taking gradients of "product" models, where
    # the function takes two iterators over the function evaluations and
    # gradients of the sub-models. Here both iterables should have length given
    # by the number of sub-models, but `giterable` will itself contain
    # iterables for each hyperparameter of the constituent models evaluate each
    # submodel.

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

    # XXX: it should now hold that M[i] is the product of every model
    # evaluation EXCEPT for the ith one.

    for Mi, grads in zip(M, giterable):
        for dM in grads:
            yield Mi*dM
