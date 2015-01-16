"""
Mean tests.
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
import operator as op

# pygp imports
import pygp.means as pm


### BASE TEST CLASS ###########################################################

# children tests should initialize a kernel and two sets of points x1 and x2 in
# their __init__ method.

class MeanTest(object):
    def __init__(self, mean):
        self.mean = mean
        rng = np.random.RandomState(0)
        ndim = 2 if isinstance(self.mean, pm.Constant) else self.mean.ndim
        self.x1 = rng.rand(5, ndim)

    def test_repr(self):
        _ = repr(self.mean)

    def test_params(self):
        params = self.mean._params()
        assert all(2 <= len(p) <= 3 for p in params)
        assert sum(p[1] for p in params) == self.mean.nhyper

    def test_copy(self):
        _ = self.mean.copy()

    def test_hyper(self):
        hyper1 = self.mean.get_hyper()
        self.mean.set_hyper(self.mean.get_hyper())
        hyper2 = self.mean.get_hyper()
        nt.assert_allclose(hyper1, hyper2)

    def test_get(self):
        _ = self.mean.get(self.x1)

    def test_grad(self):
        x = self.mean.get_hyper()
        mu = lambda x, x1: self.mean.copy(x)(x1)

        G1 = np.vstack(g for g in self.mean.grad(self.x1))
        G2 = np.array([spop.approx_fprime(x, mu, 1e-8, x1)
                       for x1 in self.x1]).T

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_gradx(self):
        try:
            G1 = self.mean.gradx(self.x1)
        except NotImplementedError:
            raise nose.SkipTest()

        m, d = self.x1.shape
        mu = self.mean

        G2 = np.array([spop.approx_fprime(x1, mu, 1e-8)
                       for x1 in self.x1]).reshape(m, d)

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)


### PER MEAN TESTS ##########################################################

class TestConstant(MeanTest):
    def __init__(self):
        MeanTest.__init__(self, pm.Constant(0.3))


class TestQuadratic1D(MeanTest):
    def __init__(self):
        MeanTest.__init__(self, pm.Quadratic(0.4, 0., 1.))


class TestQuadratic2DIso(MeanTest):
    def __init__(self):
        MeanTest.__init__(self, pm.Quadratic(-0.4, 0., 0.8, ndim=2))


class TestQuadratic2D(MeanTest):
    def __init__(self):
        MeanTest.__init__(self, pm.Quadratic(0.4, [0., 1.], [2.1, 0.9]))
