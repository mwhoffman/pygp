"""
Kernel tests.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop
import nose

# pygp imports
import pygp.kernels as pk


#==============================================================================
# base test class. children tests should initialize a kernel and two sets of
# points x1 and x2 in their __init__ method.

# pylint: disable=no-member
# pylint: disable=missing-docstring

class BaseKernelTest(object):
    def test_get(self):
        _ = self.kernel.get(self.x1, self.x2)

    def test_dget(self):
        _ = self.kernel.dget(self.x1)

    def test_params(self):
        params = self.kernel._params()
        assert all(2 <= len(p) <= 3 for p in params)
        assert sum(p[1] for p in params) == self.kernel.nhyper

    def test_copy(self):
        _ = self.kernel.copy()

    def test_hyper(self):
        K1 = self.kernel.get(self.x1, self.x2)
        K2 = self.kernel.copy(self.kernel.get_hyper()).get(self.x1, self.x2)
        nt.assert_allclose(K1, K2)

    def test_transpose(self):
        K1 = self.kernel.get(self.x1, self.x2)
        K2 = self.kernel.get(self.x2, self.x1).T
        G1 = np.array(list(self.kernel.grad(self.x1, self.x2)))
        G2 = np.array(list(self.kernel.grad(self.x2, self.x1))).swapaxes(1, 2)
        nt.assert_allclose(K1, K2)
        nt.assert_allclose(G1, G2)

    def test_self(self):
        K1 = self.kernel.get(self.x1)
        K2 = self.kernel.get(self.x1, self.x1)
        G1 = np.array(list(self.kernel.grad(self.x1)))
        G2 = np.array(list(self.kernel.grad(self.x1, self.x1)))
        nt.assert_allclose(K1, K2)
        nt.assert_allclose(G1, G2)

    def test_grad(self):
        x = self.kernel.get_hyper()
        k = lambda x, x1, x2: self.kernel.copy(x)(x1, x2)

        G1 = np.array(list(self.kernel.grad(self.x1, self.x2)))
        G2 = np.array([spop.approx_fprime(x, k, 1e-8, x1, x2)
                       for x1 in self.x1
                       for x2 in self.x2])\
            .swapaxes(0, 1)\
            .reshape(-1, self.x1.shape[0], self.x2.shape[0])

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_dgrad(self):
        g1 = list(self.kernel.dgrad(self.x1))
        g2 = [np.diag(_) for _ in self.kernel.grad(self.x1)]
        nt.assert_allclose(g1, g2)

    def test_gradx(self):
        try:
            G1 = self.kernel.gradx(self.x1, self.x2)
        except NotImplementedError:
            raise nose.SkipTest()

        m = self.x1.shape[0]
        n = self.x2.shape[0]
        d = self.x1.shape[1]
        k = self.kernel

        G2 = np.array([spop.approx_fprime(x1, k, 1e-8, x2)
                       for x1 in self.x1
                       for x2 in self.x2]).reshape(m, n, d)

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_gradxy(self):
        try:
            G1 = self.kernel.gradxy(self.x1, self.x2)
        except NotImplementedError:
            raise nose.SkipTest()

        m = self.x1.shape[0]
        n = self.x2.shape[0]
        d = self.x1.shape[1]
        g = lambda x2, x1, i: self.kernel.gradx(x1[None], x2[None])[0, 0, i]

        G2 = np.array([spop.approx_fprime(x2, g, 1e-8, x1, i)
                       for x1 in self.x1
                       for x2 in self.x2
                       for i in xrange(d)]).reshape(m, n, d, d)

        nt.assert_allclose(G1, G2, rtol=1e-6, atol=1e-6)

    def test_spectrum(self):
        try:
            W, alpha = self.kernel.sample_spectrum(100)
        except NotImplementedError:
            raise nose.SkipTest()

        assert np.isscalar(alpha)
        assert W.shape[0] == 100


#==============================================================================
# Test classes.

# set the random seed to something so that we know we're testing arbitrary
# points, but the randomness will not make the tests fail between runs due to
# different levels of accuracy for different points.
np.random.seed(0)


class TestSEARD(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.SE(0.8, [0.3, 0.4])
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestSEIso(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.SE(0.8, 0.3, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestPeriodic(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Periodic(0.5, 0.4, 0.3)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestRQARD(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.RQ(0.5, [0.4, 0.5], 0.3)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestRQIso(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.RQ(0.5, 0.4, 0.3, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternARD1(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, [0.4, 0.3], d=1)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternARD3(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, [0.4, 0.3], d=3)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternARD5(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, [0.4, 0.3], d=5)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternIso1(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, 0.4, d=1, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternIso3(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, 0.4, d=3, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestMaternIso5(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, 0.4, d=5, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestRealSum(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, 0.4, d=5, ndim=2)
        self.kernel += pk.SE(0.8, 0.3, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)


class TestRealProduct(BaseKernelTest):
    def __init__(self):
        self.kernel = pk.Matern(0.5, 0.4, d=5, ndim=2)
        self.kernel *= pk.SE(0.8, 0.3, ndim=2)
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)
