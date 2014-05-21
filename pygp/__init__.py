from .inference import BasicGP
from .plotting import gpplot
from .hyper import optimize

from . import hyper
from . import kernels
from . import likelihoods

__all__ = ['BasicGP', 'optimize', 'gpplot']
