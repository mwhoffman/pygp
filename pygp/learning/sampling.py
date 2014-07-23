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
from ..utils.models import get_params

# exported symbols
__all__ = ['sample']


#===============================================================================
# basic sampler(s) that don't know anything about GP objects.

def _slice_sample(logprob, x0, sigma=1.0, step_out=True, max_steps_out=1000):
    """
    Implementation of slice sampling taken almost directly from Snoek's
    spearmint package (with a few minor modifications).
    """
    def direction_slice(direction, x0):
        def dir_logprob(z):
            return logprob(direction*z + x0)

        upper = sigma*np.random.rand()
        lower = upper - sigma
        llh_s = np.log(np.random.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper += sigma

        while True:
            new_z = (upper - lower)*np.random.rand() + lower
            new_llh = dir_logprob(new_z)
            if np.isnan(new_llh):
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        return new_z*direction + x0

    # FIXME: I've removed how blocks work because I want to rewrite that bit. so
    # right now this samples everything as one big block.
    direction = np.random.randn(x0.shape[0])
    direction = direction / np.sqrt(np.sum(direction**2))
    return direction_slice(direction, x0)


#===============================================================================
# interface for sampling hyperparameters from a GP.

def sample(gp, priors, n, burn=0, raw=True):
    priors = dict(priors)
    active = np.ones(gp.nhyper, dtype=bool)
    logged = np.ones(gp.nhyper, dtype=bool)

    for (key, block, log) in get_params(gp):
        inactive = (key in priors) and (priors[key] is None)
        logged[block] = log
        active[block] = not inactive
        if inactive: del priors[key]
        else: priors[key] = (block, log, priors[key])

    # priors is now just a list of the form (block, log, prior).
    priors = priors.values()

    # get the initial hyperparameters and transform into the non-log space.
    hyper0 = gp.get_hyper()
    hyper0[logged] = np.exp(hyper0[logged])

    def logprob(x):
        # copy the initial hyperparameters and then assign the "active"
        # parameters that come from x.
        hyper = hyper0.copy()
        hyper[active] = x
        logprob = 0

        # compute the prior probabilities. we do this first so that if there are
        # any infs they'll be caught in the least expensive computations first.
        for block, log, prior in priors:
            logprob += prior.logprior(hyper[block])
            if np.isinf(logprob): break

        # now compute the likelihood term. note that we'll have to take the log
        # of any logspace parameters before calling set_hyper.
        if not np.isinf(logprob):
            hyper[logged] = np.log(hyper[logged])
            gp.set_hyper(hyper)
            logprob += gp.loglikelihood()

        return logprob

    # create a big list of the hyperparameters so that we can just assign to the
    # components that are active. also get an initial sample x corresponding
    # only to the active parts of hyper0.
    hypers = np.tile(hyper0, (n, 1))
    x = hyper0.copy()[active]

    # do the sampling.
    for i in xrange(n):
        x = _slice_sample(logprob, x)
        hypers[i][active] = x

    # change the logspace components back into logspace.
    hypers[:, logged] = np.log(hypers[:, logged])

    # make sure the gp gets updated to the last sampled hyperparameter.
    gp.set_hyper(hypers[-1])

    if burn > 0:
        hypers = hypers[m:].copy()

    if raw:
        return hypers
    else:
        return [gp.copy(h) for h in hypers]
