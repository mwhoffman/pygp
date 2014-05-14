"""
Perform hyperparameter sampling.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from .samplers import slice_sample

# exported symbols
__all__ = ['sample']


def sample(gp, priors, n):
    hyper0 = gp.get_hyper()
    active = np.ones(gp.nhyper, dtype=bool)

    # this just manipulates a few lists so that we transform priors into a list
    # of tuples of the form (block, log, prior) for each named prior.
    params = dict((key, (block, log)) for (key,block,log) in get_params(gp))
    priors = dict() if (priors is None) else priors
    priors = [params[key] + (prior,) for (key, prior) in priors.items()]
    del params

    # remove from the active set any block where the prior is None.
    for block, _, prior in priors:
        if prior is None:
            active[block] = False

    # get rid of these simple constraint priors.
    priors = [(b,l,p) for (b,l,p) in priors if p is not None]

    def logprob(x):
        hyper = hyper0.copy(); hyper[active] = x
        gp.set_hyper(hyper)
        nlogprob = gp.nloglikelihood()
        for block, log, prior in priors:
            nlogprob += prior.nlogprior(np.exp(hyper[block]) if log else hyper[block])
        return -nlogprob

    # store all the hyperparameters we see
    hypers = [hyper0[active]]

    for i in xrange(n):
        hypers.append(slice_sample(logprob, hypers[-1]))

    # make sure the gp gets updated to the last sampled hyperparameter.
    gp.set_hyper(hypers[-1])

    # convert hypers into an array and ditch the initial point.
    return np.asarray(hypers[1:])
