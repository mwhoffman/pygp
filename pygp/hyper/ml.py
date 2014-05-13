"""
Perform type-II maximum likelihood to fit the hyperparameters of a GP model.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.optimize as so

# exported symbols
__all__ = ['optimize']


def optimize(gp, priors=None):
    """
    Perform type-II maximum likelihood to fit GP hyperparameters.

    If given the priors object should be a dictionary mapping named parameters
    to an object which implements `prior.nloglikelihood(hyper, grad)`. If a
    parameter is mapped to the `None` value then this will be assumed fixed.

    Note: nothing is returned by this function. Instead it will modify the
    hyperparameters of the given GP object in place.
    """
    priors = dict() if (priors is None) else priors.copy()
    hyper0 = gp.get_hyper()
    active = np.ones(gp.nhyper, dtype=bool)
    blocks = dict()

    # create a dictionary which maps each parameter into its set of blocks.
    offset = 0
    for key, _, size in gp._params():
        blocks[key] = slice(offset, offset+size)
        offset += size

    # loop through the priors and delete anything that corresponds to a delta
    # prior (ie don't optimize) and remove the indices from the active set.
    for key, prior in priors.items():
        if prior is None:
            active[blocks[key]] = False
            del priors[key]

    def objective(x):
        hyper = hyper0.copy()
        hyper[active] = x
        gp.set_hyper(hyper)
        nll, dnll = gp.nloglikelihood(True)
        for key, prior in priors.items():
            block = blocks[key]
            p, dp = prior.nloglikelihood(hyper[block], True)
            nll += p
            dnll[block] += dp
        return nll, dnll[active]

    # optimize the model
    x, _, info = so.fmin_l_bfgs_b(objective, hyper0[active])

    # make sure that the gp is using the correct hypers
    hyper = hyper0.copy()
    hyper[active] = x
    gp.set_hyper(hyper)
