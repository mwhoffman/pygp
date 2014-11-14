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
from mwhutils.random import rstate

# local imports
from ..learning.sampling import sample

# exported symbols
__all__ = ['MCMC']


class MCMC(object):
    def __init__(self, model, prior, n=100, burn=100, rng=None):
        self._model = model.copy()
        self._prior = prior
        self._samples = []
        self._n = n
        self._burn = burn
        self._rng = rstate(rng)

        if self._model.ndata > 0:
            if self._burn > 0:
                sample(self._model, self._prior, self._burn, rng=self._rng)
            self._samples = sample(self._model,
                                   self._prior,
                                   self._n,
                                   raw=False,
                                   rng=self._rng)

        else:
            # FIXME: the likelihood won't play a role, so we can sample
            # directly from the prior. This of course requires the prior to
            # also be a well-defined distribution.
            pass

    def __iter__(self):
        return self._samples.__iter__()

    @property
    def ndata(self):
        return self._model.ndata

    @property
    def data(self):
        return self._model.data

    def add_data(self, X, y):
        # add the data
        nprev = self._model.ndata
        self._model.add_data(X, y)

        # if we've increased the amount of data by more than a factor two we'll
        # burn off some samples. Not sure if this is entirely necessary, but it
        # also accounts for burnin right after initial data is added.
        if self._model.ndata > 2*nprev and self._burn > 0:
            sample(self._model, self._prior, self._burn, rng=self._rng)

        # grab the samples.
        self._samples = sample(self._model,
                               self._prior,
                               self._n,
                               raw=False,
                               rng=self._rng)

    def posterior(self, X, grad=False):
        parts = map(np.array,
                    zip(*[_.posterior(X, grad) for _ in self._samples]))

        mu_, s2_ = parts[:2]
        mu = np.mean(mu_, axis=0)
        s2 = np.mean(s2_ + (mu_ - mu)**2, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.mean(dmu_, axis=0)
        Dmu = dmu_ - dmu
        ds2 = np.mean(ds2_
                      + 2 * mu_[:, :, None] * Dmu
                      - 2 * mu[None, :, None] * Dmu, axis=0)

        return mu, s2, dmu, ds2
