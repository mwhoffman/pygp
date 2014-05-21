"""
Implementations of various prior objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['Uniform']


class Uniform(object):
    def __init__(self, a, b):
        self._a = np.array(a, copy=True, ndmin=1)
        self._b = np.array(b, copy=True, ndmin=1)
        self.ndim = len(self._a)

        if len(self._a) != len(self._b):
            raise RuntimeError("bound sizes don't match")

        if np.any(self._b < self._a):
            raise RuntimeError("malformed upper/lower bounds")

    def nlogprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        for a, b, t in zip(self._a, self._b, theta):
            if (t < a) or (t > b):
                return np.inf
        return 0.0
