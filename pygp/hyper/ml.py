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


def optimize(gp):
    """
    Perform type-II maximum likelihood to fit the hyperparameters of the given
    GP object. Note that this will modify the hyperparameters of the given GP.
    """
    def objective(hyper):
        gp.set_hyper(hyper)
        return gp.nloglikelihood(True)

    # optimize the model
    hyper, _, info = so.fmin_l_bfgs_b(objective, gp.get_hyper())

    # make sure that the gp is using the correct hypers
    gp.set_hyper(hyper)

    # FIXME: every call to objective will update the gp object, but it's not
    # necessarily the case that the last call to objective will be the one with
    # the lowest objective value. we could check this however, and if
    # gp.get_hyper() is equal to hyper then we do nothing...
