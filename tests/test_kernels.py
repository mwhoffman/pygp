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

# theano imports
import theano as T
import theano.tensor as TT

# pygp imports
import pygp.kernels as pk


#===============================================================================
# base test class. instances must have kernel, kfun, dhfun, and dxfun defined
# where each of these evaluates the kernel from one element (or pair of
# elements).

class BaseKernelTest(object):
    def __init__(self):
        self.x1 = np.random.rand(5, self.kernel.ndim)
        self.x2 = np.random.rand(3, self.kernel.ndim)
        self.hyper = self.kernel.get_hyper()

    def _get(self, x1, x2):
        m = x1.shape[0]
        n = x2.shape[0]
        K = [self.kfun(self.hyper, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)]
        K = np.array(K).reshape(m, n)
        return K

    def _grad(self, x1, x2):
        m = x1.shape[0]
        n = x2.shape[0]
        G = [self.dhfun(self.hyper, x1[i], x2[j]) for (i,j) in np.ndindex(m,n)]
        G = np.array(G).T.reshape(len(G[0]), m, n)
        return G

    def test_get(self):
        K1 = self.kernel.get(self.x1, self.x2)
        K2 = self._get(self.x1, self.x2)
        nt.assert_allclose(K1, K2)

    def test_grad(self):
        G1 = np.array(list(self.kernel.grad(self.x1, self.x2)))
        G2 = self._grad(self.x1, self.x2)
        nt.assert_allclose(G1, G2)

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
# definitions of kernel functions in theano which we can then use to generate
# the derivatives.

def functionize(k, x1, x2, theta):
    kfun  = T.function([theta, x1, x2], k, mode='FAST_COMPILE')
    dhfun = T.function([theta, x1, x2], T.grad(k, theta), mode='FAST_COMPILE')
    dxfun = T.function([theta, x1, x2], T.grad(k, x1), mode='FAST_COMPILE')
    return kfun, dhfun, dxfun


def sqdist(x1, x2, ell=None):
    if ell is not None:
        x1 = x1 / ell
        x2 = x2 / ell
    return TT.dot(x1, x1) + TT.dot(x2, x2) - 2*TT.dot(x1, x2)


def dist(x1, x2, ell=None):
    return TT.sqrt(sqdist(x1, x2, ell))


def se(iso=False):
    theta = TT.vector('theta')
    sf2 = TT.exp(theta[0]*2)
    ell = TT.exp(theta[1] if iso else theta[1:])
    x1 = TT.vector('x1')
    x2 = TT.vector('x2')
    k = sf2 * TT.exp(-0.5*sqdist(x1, x2, ell))
    return k, x1, x2, theta


def periodic():
    theta = TT.vector('theta')
    sf2 = TT.exp(theta[0]*2)
    ell = TT.exp(theta[1])
    p = TT.exp(theta[2])
    x1 = TT.vector('x1')
    x2 = TT.vector('x2')
    k = sf2 * TT.exp(-2*(TT.sin(dist(x1, x2) * np.pi / p) / ell)**2)
    return k, x1, x2, theta


def rq(iso):
    theta = TT.vector('theta')
    sf2 = TT.exp(theta[0]*2)
    ell = TT.exp(theta[1] if iso else theta[1:-1])
    alpha = TT.exp(theta[-1])
    x1 = TT.vector('x1')
    x2 = TT.vector('x2')
    k = sf2 * (1 + sqdist(x1, x2, ell)/2/alpha) ** (-alpha)
    return k, x1, x2, theta


#===============================================================================
# Test classes.

class TestSEARD(BaseKernelTest):
    kfun, dhfun, dxfun = functionize(*se())
    kernel = pk.SE(0.8, [0.3, 0.4])


class TestSEIso(BaseKernelTest):
    kfun, dhfun, dxfun = functionize(*se(iso=True))
    kernel = pk.SE(0.8, 0.3, ndim=2)


# class TestPeriodic(BaseKernelTest):
#     kfun, dhfun, dxfun = functionize(*periodic())
#     kernel = pk.Periodic(0.5, 0.4, 0.3)


# class TestRQIso(BaseKernelTest):
#     kfun, dhfun, dxfun = functionize(*rq(iso=True))
#     kernel = pk.RQIso(0.5, 0.4, 0.3)
