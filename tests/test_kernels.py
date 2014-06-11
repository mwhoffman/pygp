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

    def test_kernel(self):
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
# the following function takes the definition of a kernel function in theano and
# returns functions to evaluate the kernel, its derivatives wrt the
# hyperparameters, and the derivatives wrt x1.

def functionize(k, x1, x2, theta):
    kfun  = T.function([theta, x1, x2], k, mode='FAST_COMPILE')
    dhfun = T.function([theta, x1, x2], T.grad(k, theta), mode='FAST_COMPILE')
    dxfun = T.function([theta, x1, x2], T.grad(k, x1), mode='FAST_COMPILE')
    return kfun, dhfun, dxfun


#===============================================================================
# tests for the SE kernels

def se(logsf, logell):
    x1 = TT.vector('x1')
    x2 = TT.vector('x2')
    x1_ = x1 / TT.exp(logell)
    x2_ = x2 / TT.exp(logell)
    d = TT.dot(x1_, x1_) + TT.dot(x2_, x2_) - 2*TT.dot(x1_, x2_)
    k = TT.exp(2.0*logsf - 0.5*d)
    return k, x1, x2


def seiso():
    theta = TT.vector('theta')
    logsf = theta[0]
    logell = theta[1]
    k, x1, x2 = se(logsf, logell)
    return k, x1, x2, theta


def seard():
    theta = TT.vector('theta')
    logsf = theta[0]
    logell = theta[1:]
    k, x1, x2 = se(logsf, logell)
    return k, x1, x2, theta


class TestSEARD(BaseKernelTest):
    kfun, dhfun, dxfun = functionize(*seard())
    kernel = pk.SEARD(0.8, [0.3, 0.4])


class TestSEIso(BaseKernelTest):
    kfun, dhfun, dxfun = functionize(*seiso())
    kernel = pk.SEIso(0.8, 0.3)


#===============================================================================
# tests for the periodic kernel

def periodic():
    theta = TT.vector('theta')
    x1 = TT.vector('x1')
    x2 = TT.vector('x2')

    sf2 = TT.exp(theta[0]*2)
    ell = TT.exp(theta[1])
    p = TT.exp(theta[2])

    d = TT.sqrt(TT.dot(x1, x1) + TT.dot(x2, x2) - 2*TT.dot(x1, x2)) * np.pi / p
    k = sf2 * TT.exp(-2*(TT.sin(d) / ell)**2)

    return k, x1, x2, theta


class TestPeriodic(BaseKernelTest):
    kfun, dhfun, dxfun = functionize(*periodic())
    kernel = pk.Periodic(0.5, 0.4, 0.3)
