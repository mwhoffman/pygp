"""
Tests for exact GP posterior gradients.
"""

# global imports.
import os
import numpy as np
import scipy.optimize as spop

# local imports
import pygp

# "True" values
mu = np.array([0.62386198])
s2 = np.array([0.01324406])
dmu = np.array([[-4.93999657]])
ds2 = np.array([[ 0.07108522]])

def test_posterior_grad_1d():
    """
    One-dimensional test based on the basic demo's data xy.npz.

    Testing at point xtest=[0.4], should result in the following posterior
    quantities (and gradients):

        mu = [0.62386198]
        s2 = [0.01324406]
        dmu = [[-4.93999657]]
        ds2 = [[ 0.07108522]]
    """
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)

    # find the ML parameters and sample from the posterior.
    pygp.optimize(gp)

    xtest = np.array([0.4], ndmin=2)
    posterior = gp.posterior(xtest, grad=True)

    np.testing.assert_allclose(mu, posterior[0], rtol=1e-6)
    np.testing.assert_allclose(s2, posterior[1], rtol=1e-6)
    np.testing.assert_allclose(dmu, posterior[2], rtol=1e-6)
    np.testing.assert_allclose(ds2, posterior[3], rtol=1e-6)


def test_posterior_grad_2d():
    """
    Two-dimensional test based on the Branin data braninxy.npz.

    Testing at point xtest=[4.0, 6.0], should result in the following posterior
    quantities (and gradients):

        mu = [0.62386198]
        s2 = [0.01324406]
        dmu = [[-4.93999657]]
        ds2 = [[ 0.07108522]]
    """
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'braninxy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)

    # find the ML parameters and sample from the posterior.
    pygp.optimize(gp)

    # define functions to feed into approx_fprime
    def fmu(x): x = np.array(x, ndmin=2); return gp.posterior(x, grad=False)[0]
    def fs2(x): x = np.array(x, ndmin=2); return gp.posterior(x, grad=False)[1]

    # compute posterior quantities and gradients at some test point
    xtest = np.array([-2.0, 8.5], ndmin=2)
    _, _, dmu, ds2 = gp.posterior(xtest, grad=True)

    # numerical approximation of gradients
    dmu_ = spop.approx_fprime(xtest.ravel(), fmu, 1e-8)
    ds2_ = spop.approx_fprime(xtest.ravel(), fs2, 1e-8)

    # make sure everything is close to within tolerance
    np.testing.assert_allclose(np.array(dmu_, ndmin=2),
                               dmu, rtol=1e-6)
    np.testing.assert_allclose(np.array(ds2_, ndmin=2),
                               ds2, rtol=1e-6)


def test_posterior_grad_multiple_pts_2d():
    """
    Two-dimensional test based on the Branin data braninxy.npz.

    Testing at point xtest=[4.0, 6.0], should result in the following posterior
    quantities (and gradients):

        mu = [0.62386198]
        s2 = [0.01324406]
        dmu = [[-4.93999657]]
        ds2 = [[ 0.07108522]]
    """
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'braninxy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)

    # find the ML parameters and sample from the posterior.
    pygp.optimize(gp)

    # define functions to feed into approx_fprime
    def fmu(x): x = np.array(x, ndmin=2); return gp.posterior(x, grad=False)[0]
    def fs2(x): x = np.array(x, ndmin=2); return gp.posterior(x, grad=False)[1]

    # compute posterior quantities and gradients at some test point
    xtest = np.array([[-2.0, 8.5], [9.0, 10.0], [2.0, 3.0]])

    # numerical approximation of gradients
    dmu_, ds2_ = [], []
    for xk in xtest:
        dmu_ += [spop.approx_fprime(xk, fmu, 1e-8)]
        ds2_ += [spop.approx_fprime(xk, fs2, 1e-8)]

    # make sure everything is close to within tolerance
    _, _, dmu, ds2 = gp.posterior(xtest, grad=True)
    for k, (dmuk_, ds2k_) in enumerate(zip(dmu_, ds2_)):
        np.testing.assert_allclose(dmuk_,
                                   dmu[k], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ds2k_,
                                   ds2[k], rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    test_posterior_grad_1d()
    test_posterior_grad_2d()
    test_posterior_grad_multiple_pts_2d()
