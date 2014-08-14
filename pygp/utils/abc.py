"""
Modifications to ABC to allow for additional metaclass actions.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
from abc import ABCMeta as ABCMeta_
from abc import abstractmethod

# exported symbols
__all__ = ['ABCMeta', 'abstractmethod']


class ABCMeta(ABCMeta_):
    pass
