"""
Interface to GP inference.
"""

# import the basic things by default
from . import inference
from . import kernels
from . import learning
from . import likelihoods
from . import meta
from . import priors

# import the basic things by default
from .inference import BasicGP
from .learning import optimize

# and make them available.
__all__ = ['BasicGP', 'optimize']
