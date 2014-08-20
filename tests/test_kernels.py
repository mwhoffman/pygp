"""
Kernel tests.
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
import pygp.kernels as pk


### BASE TEST CLASS ###########################################################

# children tests should initialize a kernel and two sets of points x1 and x2 in
# their __init__ method.

class KernelTest(object):
    def test_repr(self):
        _ = repr(self.kernel)

    def test_params(self):
        params = self.kernel._params()
        assert all(2 <= len(p) <= 3 for p in params)
        assert sum(p[1] for p in params) == self.kernel.nhyper

    def test_copy(self):
        _ = self.kernel.copy()

    def test_hyper(self):
        hyper1 = self.kernel.get_hyper()
        self.kernel.set_hyper(self.kernel.get_hyper())
        hyper2 = self.kernel.get_hyper()
        nt.assert_allclose(hyper1, hyper2)

    def test_get(self):
        _ = self.kernel.get(self.x1, self.x2)

    def test_dget(self):
        _ = self.kernel.dget(self.x1)

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


### REAL KERNEL TEST CLASS ####################################################

class RealKernelTest(KernelTest):
    def __init__(self, kernel):
        self.kernel = kernel
        rng = np.random.RandomState(0)
        self.x1 = rng.rand(5, self.kernel.ndim)
        self.x2 = rng.rand(3, self.kernel.ndim)

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


### PER KERNEL TESTS ##########################################################

class TestSEARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.SE(0.8, [0.3, 0.4]))


class TestSEIso(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.SE(0.8, 0.3, ndim=2))


class TestPeriodic(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Periodic(0.5, 0.4, 0.3))


class TestRQARD(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.RQ(0.5, [0.4, 0.5], 0.3))


class TestRQIso(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.RQ(0.5, 0.4, 0.3, ndim=2))


class TestMaternARD1(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, [0.4, 0.3], d=1))


class TestMaternARD3(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, [0.4, 0.3], d=3))


class TestMaternARD5(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, [0.4, 0.3], d=5))


class TestMaternIso1(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, 0.4, d=1, ndim=2))


class TestMaternIso3(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, 0.4, d=3, ndim=2))


class TestMaternIso5(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self, pk.Matern(0.5, 0.4, d=5, ndim=2))


class TestRealSum(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self,
                                pk.SE(0.8, 0.3, ndim=2) +
                                pk.SE(0.1, 0.2, ndim=2))


class TestRealProduct(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self,
                                pk.SE(0.8, 0.3, ndim=2) *
                                pk.SE(0.1, 0.2, ndim=2))


class TestRealSumProduct(RealKernelTest):
    def __init__(self):
        RealKernelTest.__init__(self,
                                pk.SE(0.8, 0.3, ndim=2) *
                                pk.SE(0.1, 0.2, ndim=2) +
                                pk.SE(0.8, 0.3, ndim=2) *
                                pk.SE(0.1, 0.2, ndim=2))


### INITIALIZATION TESTS ######################################################

# the following tests attempt to initialize a few kernels with invalid
# parameters, each of which should raise an exception.

def test_init_sum():
    k1 = pk.SE(1, 1, ndim=1)
    k2 = pk.SE(1, 1, ndim=2)
    nt.assert_raises(ValueError, op.add, k1, k2)


def test_init_product():
    k1 = pk.SE(1, 1, ndim=1)
    k2 = pk.SE(1, 1, ndim=2)
    nt.assert_raises(ValueError, op.mul, k1, k2)


def test_init_ard():
    def check_ard(Kernel, args):
        nt.assert_raises(ValueError, Kernel, *args, ndim=1)

    kernel_args = [
        (pk.SE, (1, [1, 1])),
        (pk.Matern, (1, [1, 1])),
        (pk.RQ, (1, [1, 1], 1))
    ]

    for Kernel, args in kernel_args:
        yield check_ard, Kernel, args


def test_init_matern():
    nt.assert_raises(ValueError, pk.Matern, 1, 1, d=12)
