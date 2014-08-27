"""
Objects which implement GP inference.
"""

# pylint: disable=wildcard-import
from .exact import *
from .fitc import *
from .basic import *

from . import exact
from . import fitc
from . import basic

__all__ = []
__all__ += exact.__all__
__all__ += fitc.__all__
__all__ += basic.__all__
