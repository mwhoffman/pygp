"""
Interface for latent function inference in Gaussian process models. These models
will assume that the hyperparameters are fixed and any optimization and/or
sampling of these parameters will be left to a higher-level wrapper.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.special as ss
import abc

# local imports
from ..utils.models import Parameterized

# exported symbols
__all__ = ['GPModel']


class GPModel(Parameterized):
    def __init__(self, likelihood, kernel):
        self._likelihood = likelihood
        self._kernel = kernel
        self._X = None
        self._y = None

    def add_data(self, X, y):
        X = self._kernel.transform(X)
        y = self._likelihood.transform(y)

        if self._X is None:
            self._X = X.copy()
            self._y = y.copy()
            self._update()

        elif hasattr(self, '_updateinc'):
            self._updateinc(X, y)
            self._X = np.r_[self._X, X]
            self._y = np.r_[self._y, y]

        else:
            self._X = np.r_[self._X, X]
            self._y = np.r_[self._y, y]
            self._update()

    def predict(self, X, ci=None):
        X = self._kernel.transform(X)
        mu, s2 = self._posterior(X)
        if ci is None:
            return mu, s2
        else:
            b2 = ss.erfinv(2*(1-0.5*(1-ci))-1)
            er = np.sqrt(2*b2*s2)
            return mu, mu-er, mu+er

    @abc.abstractmethod
    def _update(self):
        pass

    @abc.abstractmethod
    def _posterior(self, X, diag=True):
        pass
