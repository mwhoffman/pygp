"""
Implementations of various prior objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss

from mwhutils.random import rstate

# exported symbols
__all__ = ['Uniform', 'Gaussian', 'Gamma', 'LogNormal', 'Horseshoe']


class Uniform(object):
    def __init__(self, a, b):
        self._a = np.array(a, copy=True, ndmin=1)
        self._b = np.array(b, copy=True, ndmin=1)
        self.ndim = len(self._a)

        if len(self._a) != len(self._b):
            raise RuntimeError("bound sizes don't match")

        if np.any(self._b < self._a):
            raise RuntimeError("malformed upper/lower bounds")

    def sample(self, size=1, log=True, rng=None):
        rng = rstate(rng)
        sample = self._a + (self._b - self._a) * rng.rand(size, self.ndim)
        return np.log(sample) if log else sample

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        for a, b, t in zip(self._a, self._b, theta):
            if (t < a) or (t > b):
                return -np.inf
        return 0.0


class Gaussian(object):
    def __init__(self, mu, var):
        self._mu = np.array(mu, copy=True, ndmin=1)
        self._s2 = np.array(var, copy=True, ndmin=1)
        self.ndim = len(self._mu)
        self._log2pi = np.log(2*np.pi)

        if self._s2.ndim == 1:
            self._std = np.sqrt(self._s2)
        elif self._s2.ndim == 2:
            self._std = np.linalg.cholesky(self._s2)
        else:
            raise ValueError('Argument `var` can be at most a rank 2 array.')

    def sample(self, size=1, log=True, rng=None):
        rng = rstate(rng)
        if self._std.ndim == 1:
            sample = self._mu + self._std * rng.randn(size, self.ndim)
        elif self._s2.ndim == 2:
            sample = self._mu + np.dot(rng.randn(size, self.ndim), self._std)

        return np.log(sample) if log else sample

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)

        if self._s2.ndim == 1:
            logpdf = np.sum(np.log(self._s2))                   # log det
            logpdf += np.sum((theta - self._mu)**2 / self._s2)  # mahalanobis
            logpdf += self.ndim * self._log2pi
            logpdf *= -0.5
        elif self._s2.ndim == 2:
            logpdf = ss.multivariate_normal.logpdf(theta,
                                                   mean=self._mu,
                                                   cov=self._s2)
        return logpdf


class Gamma(object):
    def __init__(self, k, scale, min=0.):
        self._k = np.array(k, copy=True, ndmin=1)
        self._scale = np.array(scale, copy=True, ndmin=1)
        self._min = min
        self.ndim = len(self._k)

    def sample(self, size=1, log=True, rng=None):
        rng = rstate(rng)
        sample = np.vstack(self._min + rng.gamma(k, s, size=size)
                           for k, s in zip(self._k, self._scale)).T

        return np.log(sample) if log else sample

    def logprior(self, theta):
        # note the theta in this function *does not* correspond to the scale
        # parameter of a Gamma distribution which is denoted here as _scale.
        theta = np.array(theta, copy=False, ndmin=1)
        if np.any(theta <= self._min):
            return -np.inf

        logpdf = ss.gamma.logpdf(self._k,
                                 theta,
                                 scale=self._scale,
                                 loc=self._min)

        return logpdf.sum()


class LogNormal(object):
    def __init__(self, mu=0., sigma=1., min=0.):
        self._mu = np.array(mu, copy=True, ndmin=1)
        self._sigma = np.array(sigma, copy=True, ndmin=1)
        self._min = min
        self.ndim = len(self._mu)

    def sample(self, size=1, log=True, rng=None):
        rng = rstate(rng)
        sample = np.vstack(self._min + rng.lognormal(m, s, size=size)
                           for m, s in zip(self._mu, self._sigma)).T

        return np.log(sample) if log else sample

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        if np.any(theta <= self._min):
            return -np.inf

        logpdf = ss.lognorm.logpdf(theta,
                                   self._sigma,
                                   scale=np.exp(self._mu),
                                   loc=self._min)

        return logpdf.sum()


class Horseshoe(object):
    def __init__(self, scale=1., min=0.):
        self._scale = np.array(scale, copy=True, ndmin=1)
        self._min = min
        self.ndim = len(self._scale)

    def sample(self, size=1, log=True, rng=None):
        raise NotImplementedError

    def logprior(self, theta):
        theta = np.array(theta, copy=False, ndmin=1)
        if np.any(theta <= self._min):
            return -np.inf

        logpdf = np.log(np.log(1 + (self._scale / (theta - self._min))**2))

        return logpdf.sum()
