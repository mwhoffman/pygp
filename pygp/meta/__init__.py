"""
Meta-models which act like a GP object but also marginalize over the
hyperparameters.
"""

# pylint: disable=wildcard-import
from .mcmc import *

from . import mcmc

__all__ = []
__all__ += mcmc.__all__
