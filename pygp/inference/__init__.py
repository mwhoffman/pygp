"""
Objects which implement GP inference.
"""

# pylint: disable=wildcard-import
from .exact import *
from .basic import *

from . import exact
from . import basic

__all__ = []
__all__ += exact.__all__
__all__ += basic.__all__
