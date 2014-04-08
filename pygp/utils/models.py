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
__all__ = ['Parameterized']


class Parameterized:
    __metaclass__ = abc.ABCMeta
