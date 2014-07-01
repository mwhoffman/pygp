"""
Implementation of the squared-exponential kernels.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.testing as nt
import nose

# theano imports
import theano as T
import theano.tensor as TT

# pygp imports
import pygp.kernels as pk


#===============================================================================
# definitions of various kernels in theano.

# each kernel will use the same symbolic vectors for the hyperparameters and the
# inputs to the kernel. this is so we don't need to pass them around in order to
# take derivatives.
THETA = TT.vector('theta')
X1 = TT.vector('x1')
X2 = TT.vector('x2')


def se(iso=False):
    sf2 = TT.exp(THETA[0]*2)
    ell = TT.exp(THETA[1] if iso else THETA[1:])
    v = (X1-X2) / ell
    k = sf2 * TT.exp(-0.5*TT.dot(v,v))
    return k


def periodic():
    sf2 = TT.exp(THETA[0]*2)
    ell = TT.exp(THETA[1])
    p = TT.exp(THETA[2])
    v = X1-X2
    k = sf2 * TT.exp(-2*(TT.sin(TT.sqrt(TT.dot(v,v)) * np.pi / p) / ell)**2)
    return k


def rq(iso=False):
    sf2 = TT.exp(THETA[0]*2)
    ell = TT.exp(THETA[1] if iso else THETA[1:-1])
    alpha = TT.exp(THETA[-1])
    v = (X1-X2) / ell
    k = sf2 * (1 + TT.dot(v,v)/2/alpha) ** (-alpha)
    return k


def matern(d, iso=False):
    sf2 = TT.exp(THETA[0]*2)
    ell = TT.exp(THETA[1] if iso else THETA[1:])
    f = (lambda r: 1  )         if (d == 1) else \
        (lambda r: 1+r)         if (d == 3) else \
        (lambda r: 1+r*(1+r/3))
    v = (X1-X2) / ell
    r = np.sqrt(d) * TT.sqrt(TT.dot(v,v))
    k = sf2 * f(r) * TT.exp(-r)
    return k


#===============================================================================
# base test class. instances must have kernel, kfun, dhfun, and dxfun defined
# where each of these evaluates the kernel from one element (or pair of
# elements).

class BaseKernelTest(object):
    def __init__(self):
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)

    def _get(self, x1, x2):
        kfun = T.function([THETA, X1, X2], self.k, mode='FAST_COMPILE')
        theta = self.kernel.get_hyper()
        m = x1.shape[0]
        n = x2.shape[0]
        K = [kfun(theta, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)]
        K = np.array(K).reshape(m, n)
        return K

    def _grad(self, x1, x2):
        dfun = T.function([THETA, X1, X2], T.grad(self.k, THETA), mode='FAST_COMPILE')
        theta = self.kernel.get_hyper()
        m = x1.shape[0]
        n = x2.shape[0]
        N = self.kernel.nhyper
        G = [dfun(theta, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)]
        G = np.array(G).T.reshape(N, m, n)
        return G

    def _gradx(self, x1, x2):
        dfun = T.function([THETA, X1, X2], T.grad(self.k, X1), mode='FAST_COMPILE')
        theta = self.kernel.get_hyper()
        m = x1.shape[0]
        n = x2.shape[0]
        d = x1.shape[1]
        G = [dfun(theta, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)]
        G = np.array(G).reshape(m, n, d)
        return G

    def _gradxy(self, x1, x2):
        dx = T.grad(self.k, X1)
        dxy, updates = T.scan(lambda i: T.grad(dx[i], X2), sequences=TT.arange(X1.shape[0]))
        dfun = T.function([THETA, X1, X2], dxy, updates=updates, mode='FAST_COMPILE')
        theta = self.kernel.get_hyper()
        m = x1.shape[0]
        n = x2.shape[0]
        d = x1.shape[1]
        G = np.array([dfun(theta, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)])
        G = np.array(G).reshape(m, n, d, d)
        return G

    def test_copy(self):
        kernel = self.kernel.copy()
        K1 = kernel.get(self.x1, self.x2)
        K2 = self.kernel.get(self.x1, self.x2)
        nt.assert_allclose(K1, K2)

    def test_hyper(self):
        kernel = self.kernel.copy()
        kernel.set_hyper(kernel.get_hyper())
        K1 = kernel.get(self.x1, self.x2)
        K2 = self.kernel.get(self.x1, self.x2)
        nt.assert_allclose(K1, K2)

    def test_get(self):
        K1 = self.kernel.get(self.x1, self.x2)
        K2 = self._get(self.x1, self.x2)
        nt.assert_allclose(K1, K2)

    def test_grad(self):
        G1 = np.array(list(self.kernel.grad(self.x1, self.x2)))
        G2 = self._grad(self.x1, self.x2)
        nt.assert_allclose(G1, G2)

    def test_gradx(self):
        if hasattr(self.kernel, 'gradx'):
            G1 = self.kernel.gradx(self.x1, self.x2)
            G2 = self._gradx(self.x1, self.x2)
            nt.assert_allclose(G1, G2)
        else:
            raise nose.SkipTest()

    def test_gradxy(self):
        if hasattr(self.kernel, 'gradxy'):
            G1 = self.kernel.gradxy(self.x1, self.x2)
            G2 = self._gradxy(self.x1, self.x2)
            nt.assert_allclose(G1, G2)
        else:
            raise nose.SkipTest()

    def test_dget(self):
        k1 = self.kernel.dget(self.x1)
        k2 = np.diag(self.kernel.get(self.x1))
        nt.assert_allclose(k1, k2)

    def test_dgrad(self):
        g1 = list(self.kernel.dgrad(self.x1))
        g2 = map(np.diag, list(self.kernel.grad(self.x1)))
        nt.assert_allclose(g1, g2)

    def test_transpose(self):
        K1 = self.kernel.get(self.x1, self.x2)
        K2 = self.kernel.get(self.x2, self.x1).T
        nt.assert_allclose(K1, K2)
        G1 = np.array(list(self.kernel.grad(self.x1, self.x2)))
        G2 = np.array(list(self.kernel.grad(self.x2, self.x1))).swapaxes(1,2)
        nt.assert_allclose(G1, G2)

    def test_self(self):
        K1 = self.kernel.get(self.x1)
        K2 = self.kernel.get(self.x1, self.x1)
        nt.assert_allclose(K1, K2)
        G1 = np.array(list(self.kernel.grad(self.x1)))
        G2 = np.array(list(self.kernel.grad(self.x1, self.x1)))
        nt.assert_allclose(G1, G2)


#===============================================================================
# Test classes.

class TestSEARD(BaseKernelTest):
    k = se()
    kernel = pk.SE(0.8, [0.3, 0.4])


class TestSEIso(BaseKernelTest):
    k = se(iso=True)
    kernel = pk.SE(0.8, 0.3, ndim=2)


class TestPeriodic(BaseKernelTest):
    k = periodic()
    kernel = pk.Periodic(0.5, 0.4, 0.3)


class TestRQARD(BaseKernelTest):
    k = rq()
    kernel = pk.RQ(0.5, [0.4, 0.5], 0.3)


class TestRQIso(BaseKernelTest):
    k = rq(iso=True)
    kernel = pk.RQ(0.5, 0.4, 0.3, ndim=2)


class TestMaternARD1(BaseKernelTest):
    k = matern(1)
    kernel = pk.Matern(0.5, [0.4, 0.3], d=1)


class TestMaternARD3(BaseKernelTest):
    k = matern(3)
    kernel = pk.Matern(0.5, [0.4, 0.3], d=3)


class TestMaternARD5(BaseKernelTest):
    k = matern(5)
    kernel = pk.Matern(0.5, [0.4, 0.3], d=5)


class TestMaternIso1(BaseKernelTest):
    k = matern(1, iso=True)
    kernel = pk.Matern(0.5, 0.4, d=1, ndim=2)


class TestMaternIso3(BaseKernelTest):
    k = matern(3, iso=True)
    kernel = pk.Matern(0.5, 0.4, d=3, ndim=2)


class TestMaternIso5(BaseKernelTest):
    k = matern(5, iso=True)
    kernel = pk.Matern(0.5, 0.4, d=5, ndim=2)
