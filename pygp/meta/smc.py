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

# exported symbols
__all__ = ['SMC']


class SMC(object):
    def __init__(self, model, prior, n=100, mcmc=0):
        self._prior = prior
        self._n = n
        self._mcmc = mcmc   # number of mcmc steps in particle propagation

        self._samples = self._prior_sampling(model, prior)
        self._logweights = np.zeros(n) - np.log(n)

        if model.ndata > 0:
            self.add_data(*model.data)

    def __iter__(self):
        return self._samples.__iter__()

    def _prior_sampling(self, model, priors):
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
        hypers = np.zeros((self._n, model.nhyper))
        for (block, log, prior) in priors:
            hypers[:, block] = prior.sample(self._n, log=log)

        # TODO -- Bobak: The following functionality should be provided by the
        # base model e.g. in the form of a model.reset() function. model.copy()
        # could be made to take an optional nodata=False kwarg that copies the
        # object without its data. In which case the following lines can simply
        # be replaced with:
        #
        # return [model.copy(h, nodata=True) for h in hypers]
        samples = [model.copy(h) for h in hypers]
        for sample in samples:
            sample._X = None
            sample._y = None
        return samples

    @property
    def ndata(self):
        return self._samples[-1].ndata

    @property
    def data(self):
        return self._samples[-1].data

    @property
    def ess(self):
        w2 = np.exp(2 * (self._logweights - logsumexp(self._logweights)))
        return 1.0 / w2.sum()

    def add_data(self, X, y):
        for (xi, yi) in zip(X, y):
            # compute likelihood of previous data
            loglikes = [model.loglikelihood() for model in self._samples]

            # add data
            for model in self._samples:
                model.add_data(xi, yi)

            # resample if effective sample size is less than N/2
            if self.ess < self._n / 2:
                weights = np.exp(self._logweights - logsumexp(self._logweights))
                idx = np.random.choice(self._n, self._n, p=weights)
                loglikes = [loglikes[i] for i in idx]
                self._samples = [self._samples[i] for i in idx]
                self._logweights = np.zeros(self._n) - np.log(self._n)

            # incremental weights are given by Eq. 31 in (Del Moral et al., 2006)
            # Note: according to Eq. 31 this likelihood has to be computed
            # before propagation but after data is added.
            logratio = [model.loglikelihood() - loglike_
                        for (model, loglike_) in zip(self._samples, loglikes)]
            self._logweights += np.array(logratio)

            # propagate particles according to MCMC kernel as per Eq. 30 in
            # (Del Moral et al., 2006)
            self._samples = [sample(model, self._prior, self._mcmc+1, raw=False)[-1]
                             for model in self._samples]


    def posterior(self, X, grad=False):
        parts = map(np.array, zip(*[_.posterior(X, grad) for _ in self._samples]))
        weights = np.exp(self._logweights - logsumexp(self._logweights))

        mu_, s2_ = parts[:2]
        mu = np.average(mu_, weights=weights, axis=0)
        s2 = np.average(s2_ + (mu_ - mu)**2, weights=weights, axis=0)

        if not grad:
            return mu, s2

        dmu_, ds2_ = parts[2:]
        dmu = np.average(dmu_, weights=weights, axis=0)
        Dmu = dmu_ - dmu
        ds2 = np.average(ds2_ + 2 * mu_[:, :, None] * Dmu
                         - 2 * mu[None, :, None] * Dmu, weights=weights, axis=0)

        return mu, s2, dmu, ds2
