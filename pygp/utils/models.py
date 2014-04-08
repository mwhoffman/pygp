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


class Parameterized:
    """
    Interface for objects that are parameterized by some set of hyperparameters.
    """
    __metaclass__ = abc.ABCMeta


class Printable:
    """
    Interface for objects which can be pretty-printed as a function of their
    hyperparameters.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _params(self):
        pass

    def __repr__(self):
        substrings = ['%s=%s' % kv for kv in self._params()]
        return self.__class__.__name__ + '(' + ', '.join(substrings) + ')'
