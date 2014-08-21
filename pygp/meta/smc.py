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
from ..learning.sampling import sample

# exported symbols
__all__ = ['SMC']


class SMC(object):
    def __init__(self, model, prior, n=100, burn=0):
        self._model = model
        self._prior = prior
        # FIXME -- Bobak: the following should initialize according to prior
        self._samples = [model.copy() for _ in xrange(n)]
        self._weights = np.ones(n) / n
        self._n = n
        self._burn = burn   # every particle will burn before propagating

        if self._model.ndata > 0:
            self._update()

        else:
            # FIXME: the likelihood won't play a role, so we can sample directly
            # from the prior. This of course requires the prior to also be
            # a well-defined distribution.
            pass

    def __iter__(self):
        return self._samples.__iter__()

    def _update(self, n=1):
        # weights are given on page 5 of (Gramacy & Polson, 2010) derived from Eq. 31
        # in (Del Moral et al., 2006)
        self._weights *= np.array([model.loglikelihood(n=n)
                                   for model in self._samples])

        weights = self._weights / self._weights.sum()
        # resample if effective sample size is less than N/2
        if np.sum(weights ** 2) * weights.shape[0] > 2:
            self._resample()

        # propagate particles according to MCMC kernel (1 step) as per Eq. 30 in
        # (Del Moral et al., 2006)
        self._samples = [sample(model, self._prior, self._burn+1, raw=False)[-1]
                         for model in self._samples]

        self._model = self._samples[-1]

    def _resample(self):
        n = self._weights.shape[0]
        weights = self._weights / self._weights.sum()
        self._samples = np.random.choice(self._samples, size=n, p=weights)
        self._weights = np.ones(n) / n

    @property
    def ndata(self):
        return self._model.ndata

    @property
    def data(self):
        return self._model.data

    def add_data(self, X, y):
        for model in self._samples:
            model.add_data(X, y)
        self._update(n=X.shape[0])

    def posterior(self, X, grad=False):
        parts = map(np.array, zip(*[_.posterior(X, grad) for _ in self._samples]))
        weights = self._weights / self._weights.sum()

        mu_, s2_ = parts[:2]
        mu = np.average(mu_, weights=weights, axis=0)
        s2 = np.average(s2_ + (mu_ - mu)**2, weights=weights, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.average(dmu_, weights=weights, axis=0)
        Dmu = dmu_ - dmu
        ds2 = np.average(ds2_ + 2 * mu_[:,:,None] * Dmu
                         - 2 * mu [None,:,None] * Dmu, weights=weights, axis=0)

        return mu, s2, dmu, ds2
