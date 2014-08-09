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
__all__ = ['MCMC']


class MCMC(object):
    def __init__(self, model, prior, n=100, burn=0):
        self._model = model
        self._prior = prior
        self._samples = []
        self._n = n
        self._burn = burn

        if self._model.ndata > 0:
            self._update()

        else:
            # FIXME: the likelihood won't play a role, so we can sample directly
            # from the prior. This of course requires the prior to also be
            # a well-defined distribution.
            pass

    def __iter__(self):
        return self._samples.__iter__()

    def _update(self):
        self._samples = sample(self._model, self._prior, self._n, self._burn, raw=False)
        self._model = self._samples[-1]

    @property
    def ndata(self):
        return self._model.ndata

    @property
    def data(self):
        return self._model.data

    def add_data(self, X, y):
        self._model.add_data(X, y)
        self._update()

    def posterior(self, X, grad=False):
        parts = map(np.array, zip(*[_.posterior(X, grad) for _ in self._samples]))

        mu_, s2_ = parts[:2]
        mu = np.mean(mu_, axis=0)
        s2 = np.mean(s2_ + (mu_ - mu)**2, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.mean(dmu_, axis=0)
        ds2 = np.mean(ds2_ + 2*dmu_ + 2*dmu - 2*mu_[:,:,None]*dmu[None]
                                            - 2*mu [None,:,None]*dmu_,
                                            axis=0)

        return mu, s2, dmu, ds2
