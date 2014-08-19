"""
Objects implementing likelihoods.
"""

# pylint: disable=wildcard-import
from .gaussian import *

from . import gaussian

__all__ = []
__all__ += gaussian.__all__
