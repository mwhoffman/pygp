"""
Objects which implement the mean interface.
"""

# pylint: disable=wildcard-import
from .constant import *
from .quadratic import *
from .hinge import *

from . import constant
from . import quadratic
from . import hinge

__all__ = []
__all__ += constant.__all__
__all__ += quadratic.__all__
__all__ += hinge.__all__