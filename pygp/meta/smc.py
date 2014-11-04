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
from scipy.misc import logsumexp

# local imports
from ..learning.sampling import sample
from ..utils.models import get_params
from ..utils.random import rstate

# exported symbols
__all__ = ['SMC']


def _sample_prior(model, priors, n, rng=None):
    rng = rstate(rng)

    # unpack priors
    # TODO -- Bobak: This snippet is copied from learning/sampling.py
    # and should probably be put into a Prior base class.
    priors = dict(priors)
    active = np.ones(model.nhyper, dtype=bool)
    logged = np.ones(model.nhyper, dtype=bool)

    for (key, block, log) in get_params(model):
        inactive = (key in priors) and (priors[key] is None)
        logged[block] = log
        active[block] = not inactive
        if inactive:
            del priors[key]
        else:
            priors[key] = (block, log, priors[key])
    priors = priors.values()

    # sample hyperparameters from prior
    hypers = np.tile(model.get_hyper(), (n, 1))
    for (block, log, prior) in priors:
        try:
            hypers[:, block] = prior.sample(n, log=log, rng=rng)
        except NotImplementedError:
            pass

    return hypers


class SMC(object):
    def __init__(self, model, prior, n=100, rng=None):
        self._prior = prior
        self._n = n
        self._rng = rstate(rng)

        # we won't add any data unless the model already has it.
        data = None

        if model.ndata > 0:
            data = model.data
            model = model.copy()
            model.reset()

        self._samples = [model.copy(h)
                         for h in _sample_prior(model, prior, n, rng=self._rng)]
        self._logweights = np.zeros(n) - np.log(n)
        self._loglikes = np.zeros(n)

        if data is not None:
            self.add_data(data[0], data[1])

    def __iter__(self):
        return self._samples.__iter__()

    @property
    def ndata(self):
        return self._samples[-1].ndata

    @property
    def data(self):
        return self._samples[-1].data

    def add_data(self, X, y):
        X = self._samples[0]._kernel.transform(X)
        y = self._samples[0]._likelihood.transform(y)

        for (xi, yi) in zip(X, y):
            # resample if effective sample size is less than N/2
            if -logsumexp(2*self._logweights) < np.log(self._n/2):
                # FIXME: can use a better resampling strategy here. ie,
                # stratified, etc.
                p = np.exp(self._logweights)
                idx = self._rng.choice(self._n, self._n, p=p)
                self._samples = [self._samples[i].copy() for i in idx]
                self._logweights = np.zeros(self._n) - np.log(self._n)
                self._loglikes = self._loglikes[idx]

            # add data
            for model in self._samples:
                model.add_data(xi, yi)

            # we will propose new hyperparameters using an MCMC kernel, which
            # corresponds to Eqs. 30--31 of (Del Moral et al, 2006). To compute
            # the incremental weights we just need the loglikelihoods before
            # and after adding the new data but before propagating the
            # particles.

            # self._loglikes already contains the loglikelihood before adding
            # the data, so what follows computes it after adding the data.
            loglikes = np.fromiter((model.loglikelihood()
                                    for model in self._samples), float)

            # update and normalize the weights.
            self._logweights += loglikes - self._loglikes
            self._logweights -= logsumexp(self._logweights)

            # propagate the particles.
            for model in self._samples:
                sample(model, self._prior, 1, rng=self._rng)

            # update the loglikelihoods given the new samples.
            self._loglikes = np.fromiter((model.loglikelihood()
                                          for model in self._samples), float)

    def posterior(self, X, grad=False):
        parts = [_.posterior(X, grad) for _ in self._samples]
        parts = [np.array(_) for _ in zip(*parts)]

        weights = np.exp(self._logweights)

        mu_, s2_ = parts[:2]
        mu = np.average(mu_, weights=weights, axis=0)
        s2 = np.average(s2_ + (mu_ - mu)**2, weights=weights, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.average(dmu_, weights=weights, axis=0)

        Dmu = dmu_ - dmu
        ds2 = np.average(ds2_
                         + 2 * mu_[:, :, None] * Dmu
                         - 2 * mu[None, :, None] * Dmu,
                         weights=weights, axis=0)

        return mu, s2, dmu, ds2
