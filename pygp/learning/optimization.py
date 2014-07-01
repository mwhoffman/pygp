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

# local imports
from ..utils.models import get_params

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
    hyper0 = gp.get_hyper()
    active = np.ones(gp.nhyper, dtype=bool)

    # this just manipulates a few lists so that we transform priors into a list
    # of tuples of the form (block, log, prior) for each named prior.
    params = dict((key, (block, log)) for (key,block,log) in get_params(gp))
    priors = dict() if (priors is None) else priors
    priors = [params[key] + (prior,) for (key, prior) in priors.items()]
    del params

    # remove from the active any block where the prior is None.
    for block, _, prior in priors:
        if prior is None:
            active[block] = False

    # get rid of these simple constraint priors.
    priors = [(b,l,p) for (b,l,p) in priors if p is not None]

    # FIXME: right now priors won't work because I am not dealing with the any
    # of the log transformed components.
    assert len(priors) == 0

    def objective(x):
        hyper = hyper0.copy(); hyper[active] = x
        gp.set_hyper(hyper)
        nll, dnll = gp.nloglikelihood(True)
        return nll, dnll[active]

    # optimize the model
    x, _, info = so.fmin_l_bfgs_b(objective, hyper0[active])

    # make sure that the gp is using the correct hypers
    hyper = hyper0.copy()
    hyper[active] = x
    gp.set_hyper(hyper)


# FIXME: the following code can be used for optimizing wrt some prior, but this
# needs to be modified to account for the log term.
#-------------------------------------------------------------------------------
# for key, prior in priors.items():
#     block = blocks[key]
#     p, dp = prior.nloglikelihood(hyper[block], True)
#     nll += p
#     dnll[block] += dp
