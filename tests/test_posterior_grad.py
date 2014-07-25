"""
Test for exact GP posterior gradients.

One-dimensional test based on the basic demo's data xy.npz.

Testing at point xtest=[0.4], should result in the following posterior
quantities (and gradients):

    mu = [0.62386198]
    s2 = [0.01324406]
    dmu = [[-4.93999657]]
    ds2 = [[ 0.07108522]]
"""

# global imports.
import os
import numpy as np

# local imports
import pygp

# "True" values
mu = np.array([0.62386198])
s2 = np.array([0.01324406])
dmu = np.array([[-4.93999657]])
ds2 = np.array([[ 0.07108522]])

def test_posterior_grad():
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

if __name__ == '__main__':
    test_posterior_grad()