"""
Tests of inference methods.
"""

# pylint: disable=no-member
# pylint: disable=missing-docstring

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop
import nose

# local imports
import pygp


### BASE TEST CLASS ###########################################################

class InferenceTest(object):
    def test_repr(self):
        _ = repr(self.gp)

    def test_params(self):
        _ = self.gp._params()

    def test_data(self):
        _ = self.gp.data

    def test_copy(self):
        _ = self.gp.copy()

    def test_hyper(self):
        hyper1 = self.gp.get_hyper()
        self.gp.set_hyper(self.gp.get_hyper())
        hyper2 = self.gp.get_hyper()
        nt.assert_allclose(hyper1, hyper2)

    def test_add_data(self):
        # add additional data.
        gp1 = self.gp.copy()
        gp1.add_data(self.X, self.y)

        # add additional data but make sure we don't do so incrementally.
        updateinc = pygp.inference._base.GP._updateinc
        gp2 = self.gp.copy()
        gp2._updateinc = lambda X, y: updateinc(gp2, X, y)
        gp2.add_data(self.X, self.y)

        # make sure the posteriors match.
        p1 = gp1.posterior(self.X)
        p2 = gp2.posterior(self.X)
        nt.assert_allclose(p1, p2)

    def test_sample(self):
        _ = self.gp.sample(self.X, m=2, latent=False)

    def test_sample_fourier(self):
        _ = self.gp.sample_fourier(10)

    def test_loglikelihood(self):
        x = self.gp.get_hyper()
        f = lambda x: self.gp.copy(x).loglikelihood()
        _, g1 = self.gp.loglikelihood(grad=True)
        g2 = spop.approx_fprime(x, f, 1e-8)

        # slightly lesser gradient tolerance. mostly due to FITC.
        nt.assert_allclose(g1, g2, rtol=1e-5, atol=1e-5)


### TEST CLASS FOR REAL-VALUED INPUTS #########################################

class RealTest(InferenceTest):
    def __init__(self, gp):
        # create some data.
        rng = np.random.RandomState(0)
        X = rng.rand(10, gp._kernel.ndim)
        y = gp._likelihood.sample(rng.rand(10), rng)

        # create a gp.
        self.gp = gp
        self.gp.add_data(X, y)

        # new set of points to predict at.
        self.X = rng.rand(10, gp._kernel.ndim)
        self.y = gp._likelihood.sample(rng.rand(10), rng)

    def test_posterior_mu(self):
        f = lambda x: self.gp.posterior(x[None])[0]
        G1 = self.gp.posterior(self.X, grad=True)[2]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_posterior_s2(self):
        f = lambda x: self.gp.posterior(x[None])[1]
        G1 = self.gp.posterior(self.X, grad=True)[3]
        G2 = np.array([spop.approx_fprime(x, f, 1e-8) for x in self.X])
        nt.assert_allclose(G1, G2, rtol=1e-5, atol=1e-5)


### PER INFERENCE METHOD TESTS ################################################

class TestExact(RealTest):
    def __init__(self):
        likelihood = pygp.likelihoods.Gaussian(1)
        kernel = pygp.kernels.SE(1, 1, ndim=2)
        gp = pygp.inference.ExactGP(likelihood, kernel, 0.0)
        RealTest.__init__(self, gp)


class TestFITC(RealTest):
    def __init__(self):
        rng = np.random.RandomState(1)
        likelihood = pygp.likelihoods.Gaussian(1)
        kernel = pygp.kernels.SE(1, 1, ndim=2)
        gp = pygp.inference.FITC(likelihood, kernel, rng.rand(10, kernel.ndim))
        RealTest.__init__(self, gp)
