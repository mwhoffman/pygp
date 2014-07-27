"""
Meta models which take care of hyperparameter marginalization whenever data is
added.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from .sampling import sample

# exported symbols
__all__ = []


class SampledGP(object):
    def __init__(self, gp, priors, n=100, burn=0):
        self._gp = gp
        self._priors = priors
        self._samples = []
        self._n = n
        self._burn = burn

    def _update(self):
        self._samples = sample(self._gp, self._priors, self._n, self._burn, raw=False)
        self._gp = self._samples[-1]

    def add_data(self, X, y):
        self._gp.add_data(X, y)
        self._update()

    def posterior(self, *args, **kwargs):
        parts = zip(*[gp.posterior(*args, **kwargs) for gp in self._samples])
        return tuple([np.mean(part, axis=0) for part in parts])
