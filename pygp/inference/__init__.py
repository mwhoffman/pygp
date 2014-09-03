"""
Objects which implement GP inference.
"""

# pylint: disable=wildcard-import
from .exact import *
from .fitc import *
from .basic import *
from .nystrom import *

from . import exact
from . import fitc
from . import basic
from . import nystrom

__all__ = []
__all__ += exact.__all__
__all__ += fitc.__all__
__all__ += basic.__all__
__all__ += nystrom.__all__
