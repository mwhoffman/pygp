"""
Interfaces for parameterized objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import abc

# exported symbols
__all__ = ['Parameterized', 'Printable']


class Parameterized(object):
    """
    Interface for objects that are parameterized by some set of hyperparameters.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _params(self):
        pass

    @abc.abstractmethod
    def get_hyper(self):
        pass

    @abc.abstractmethod
    def set_hyper(self, hyper):
        pass


class Printable(object):
    """
    Interface for objects which can be pretty-printed as a function of their
    hyperparameters.
    """
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        hyper = self.get_hyper()
        substrings = []
        offset = 0
        for key, transform, size in self._params():
            val = hyper[offset:offset+size] if (size > 1) else hyper[offset]
            if transform == 'log':
                val = np.exp(val)
            substrings += ['%s=%s' % (key, val)]
            offset += size
        return self.__class__.__name__ + '(' + ', '.join(substrings) + ')'
