"""
Interfaces for parameterized objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import copy

# ABC imports.
from mwhutils.abc import ABCMeta, abstractmethod

# exported symbols
__all__ = ['Parameterized', 'printable', 'dot_params', 'get_params']


class Parameterized(object):
    """
    Interface for objects that are parameterized by some set of
    hyperparameters.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _params(self):
        """
        Define the set of parameters for the model. This should return a list
        of tuples of the form `(name, size, islog)`. If only a 2-tuple is given
        then islog will be assumed to be `True`.
        """
        pass

    @abstractmethod
    def get_hyper(self):
        """Return a vector of model hyperparameters."""
        pass

    @abstractmethod
    def set_hyper(self, hyper):
        """Set the model hyperparameters to the given vector."""
        pass

    def copy(self, hyper=None):
        """
        Copy the model. If `hyper` is given use this vector to immediately set
        the copied model's hyperparameters.
        """
        model = copy.deepcopy(self)
        if hyper is not None:
            model.set_hyper(hyper)
        return model


def printable(cls):
    """
    Decorator which marks classes as being able to be pretty-printed as a
    function of their hyperparameters. This decorator defines a __repr__ method
    for the given class which uses the class's `get_hyper` and `_params`
    methods to print it.
    """
    def _repr(obj):
        """Represent the object as a function of its hyperparameters."""
        hyper = obj.get_hyper()
        substrings = []
        for key, block, log in get_params(obj):
            val = hyper[block]
            val = val[0] if (len(val) == 1) else val
            val = np.exp(val) if log else val
            substrings += ['%s=%s' % (key, val)]
        return obj.__class__.__name__ + '(' + ', '.join(substrings) + ')'
    cls.__repr__ = _repr
    return cls


# FIXME: it's unclear how useful dot_params is. This might be replaced.

def dot_params(ns, params):
    """
    Extend a param tuple with a 'namespace'. IE prepend the key string with ns
    plus a dot.
    """
    return [("%s.%s" % (ns, p[0]),) + p[1:] for p in params]


# FIXME: the get_params function is kind of a hack in order to allow for
# simpler definitions of the _params() method. This should probably be
# replaced.

def get_params(obj):
    """
    Helper function which translates the values returned by _params() into
    something more meaningful.
    """
    offset = 0
    for param in obj._params():
        key = param[0]
        size = param[1]
        block = slice(offset, offset+size)
        log = (len(param) < 3) or param[2]
        offset += size
        yield key, block, log
