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
import scipy.linalg as sla
import scipy.special as ss
import abc

# local imports
from ..utils.models import Parameterized, dot_params
from ..utils.random import rstate
from ._fourier import FourierSample

# exported symbols
__all__ = ['GP']


class GP(Parameterized):
    def __init__(self, likelihood, kernel):
        self._likelihood = likelihood
        self._kernel = kernel
        self._X = None
        self._y = None

        self.nhyper = self._likelihood.nhyper + \
                      self._kernel.nhyper

    def __repr__(self):
        models = [repr(self._likelihood),
                  repr(self._kernel)]

        string = self.__class__.__name__ + '('
        joiner = ',\n' + (len(string) * ' ')
        string += joiner.join(models) + ')'

        return string

    def _params(self):
        params =  dot_params('like', self._likelihood._params())
        params += dot_params('kern', self._kernel._params())
        return params

    def get_hyper(self):
        # NOTE: if subclasses define any "inference" hyperparameters they can
        # implement their own get/set methods and call super().
        return np.r_[self._likelihood.get_hyper(),
                     self._kernel.get_hyper()]

    def set_hyper(self, hyper):
        a = 0
        for model in [self._likelihood, self._kernel]:
            model.set_hyper(hyper[a:a+model.nhyper])
            a += model.nhyper
        self._update()

    @property
    def ndata(self):
        return 0 if (self._X is None) else self._X.shape[0]

    @property
    def data(self):
        return (self._X, self._y)

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

    def sample(self, X, n=None, latent=True, rng=None):
        X = self._kernel.transform(X)
        flatten = (n is None)
        n = 1 if flatten else n
        p = len(X)

        # if a seed or instantiated RandomState is given use that, otherwise use
        # the global object.
        rng = rstate(rng)

        # add a tiny amount to the diagonal to make the cholesky of Sigma stable
        # and then add this correlated noise onto mu to get the sample.
        mu, Sigma = self._posterior(X)
        Sigma += 1e-10 * np.eye(p)
        f = mu[None] + np.dot(rng.normal(size=(n,p)), sla.cholesky(Sigma))

        if not latent:
            f = self._likelihood.sample(f.ravel(), rng).reshape(n,p)

        return f.ravel() if flatten else f

    def sample_fourier(self, N, rng=None):
        """
        Approximately sample a function from the GP using a fourier-basis
        expansion with N bases. See the documentation on `FourierSample` for
        details on the returned function object.
        """
        return FourierSample(N, self._likelihood, self._kernel, self._X, self._y, rng)

    @abc.abstractmethod
    def _update(self):
        pass

    @abc.abstractmethod
    def _posterior(self, X):
        pass

    @abc.abstractmethod
    def posterior(self, X, grad=False):
        pass

    @abc.abstractmethod
    def loglikelihood(self):
        pass
