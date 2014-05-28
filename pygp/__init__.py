# import the important sub-packages.
from . import hyper
from . import kernels
from . import likelihoods

# import the basic things by default
from .inference import BasicGP
from .hyper import optimize
from .plotting import gpplot

# and make them available.
__all__ = ['BasicGP', 'optimize', 'gpplot']
