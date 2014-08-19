"""
Tests for exact GP posterior gradients.
"""

# global imports.
import os
import numpy as np
import scipy.optimize as spop
import numpy.testing as nt

# local imports
import pygp
import pygp.utils.random as pgr


def test_posterior_grad():
    """
    Two-dimensional test of posterior grad.
    """

    # create the GP model.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1, ndim=2)

    # sample some data from the model.
    rng = pgr.rstate(0)
    X = rng.rand(20, 2)
    y = gp.sample(X, latent=False, rng=rng)

    # add it back in.
    gp.add_data(X, y)

    # define functions to feed into approx_fprime. These take a single point so
    # we have to use the None-index to "vectorize" it.
    fmu = lambda x: gp.posterior(x[None], grad=False)[0]
    fs2 = lambda x: gp.posterior(x[None], grad=False)[1]

    # get some test points
    xtest = rng.rand(20,2)

    # numerical approximation of gradients
    dmu_, ds2_ = [], []
    for xk in xtest:
        dmu_ += [spop.approx_fprime(xk, fmu, 1e-8)]
        ds2_ += [spop.approx_fprime(xk, fs2, 1e-8)]

    # collect the arrays
    dmu_ = np.array(dmu_)
    ds2_ = np.array(ds2_)

    # get the real gradients.
    _, _, dmu, ds2 = gp.posterior(xtest, grad=True)

    # make sure everything is close to within tolerance
    nt.assert_allclose(dmu, dmu_, rtol=1e-6, atol=1e-6)
    nt.assert_allclose(ds2, ds2_, rtol=1e-6, atol=1e-6)
