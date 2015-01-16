"""
Unit tests for different acquisition functions. This mainly tests that the
gradients of each acquisition function are computed correctly.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

# local imports
import pygp
import pygp.meta as meta
import pygp.priors as priors


class BaseMetaTest(object):
    def __init__(self):
        ndim = 2
        prior = {
            'sn':  priors.Uniform(0.01, 1.0),
            'sf':  priors.Uniform(0.01, 5.0),
            'ell': priors.Uniform([0.01]*ndim, [1.0]*ndim),
            'bias':  priors.Uniform([-2.0]*ndim, [2.0]*ndim)}

        # create the model.
        model = pygp.BasicGP(0.5, 1, [1]*ndim)
        model = self.MetaModel(model, prior, n=10, burn=0)

        # randomly generate some data and add it.
        rng = np.random.RandomState(0)
        X = rng.rand(10, ndim)
        y = rng.rand(10)
        model.add_data(X, y)

        self.model = model
        self.X = rng.rand(10, ndim)

    def test_grad_mu(self):
        _, _, dmu, _ = self.model.posterior(self.X, grad=True)
        fmu = lambda x: self.model.posterior(x[None], grad=True)[0][0]
        dmu_ = np.array([spop.approx_fprime(x, fmu, 1e-8) for x in self.X])
        nt.assert_allclose(dmu, dmu_, rtol=1e-6, atol=1e-6)

    def test_grad_s2(self):
        _, _, _, ds2 = self.model.posterior(self.X, grad=True)
        fs2 = lambda x: self.model.posterior(x[None], grad=True)[1][0]
        ds2_ = np.array([spop.approx_fprime(x, fs2, 1e-8) for x in self.X])
        nt.assert_allclose(ds2, ds2_, rtol=1e-6, atol=1e-6)


class TestMCMC(BaseMetaTest):
    MetaModel = meta.MCMC
