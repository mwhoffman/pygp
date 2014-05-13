"""
Implementation of the Gaussian likelihood model.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import RealLikelihood
from ..utils.models import Printable

# exported symbols
__all__ = ['Gaussian']


class Gaussian(RealLikelihood, Printable):
    def __init__(self, sigma):
        self._logsigma = np.log(float(sigma))
        self.nhyper = 1

    def _params(self):
        return [
            ('sigma', 'log', 1),]

    def get_hyper(self):
        return np.r_[self._logsigma]

    def set_hyper(self, hyper):
        self._logsigma = hyper[0]
