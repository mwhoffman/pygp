"""
Perform hyperparameter sampling.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

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
        hypers.append(_slice_sample(logprob, hypers[-1]))

    # make sure the gp gets updated to the last sampled hyperparameter.
    gp.set_hyper(hypers[-1])

    # convert hypers into an array and ditch the initial point.
    return np.asarray(hypers[1:])
