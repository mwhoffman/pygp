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
__all__ = ['Parameterized', 'Printable', 'dot_params', 'get_params']


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
        for key, block, log in get_params(self):
            val = hyper[block]
            val = val[0] if (len(val) == 1) else val
            val = np.exp(val) if log else val
            substrings += ['%s=%s' % (key, val)]
        return self.__class__.__name__ + '(' + ', '.join(substrings) + ')'


def dot_params(ns, params):
    """
    Extend a param tuple with a 'namespace'. IE prepend the key string with ns
    plus a dot.
    """
    return [("%s.%s" % (ns, p[0]),) + p[1:] for p in params]


def get_params(obj):
    offset = 0
    for param in obj._params():
        key = param[0]
        size = param[1]
        block = slice(offset, offset+size)
        log = not(len(param) > 2 and param[2])
        offset += size
        yield key, block, log
