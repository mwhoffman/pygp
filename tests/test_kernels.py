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
    def test_kernel(self):
        x1 = np.random.rand(5, self.kernel.ndim)
        x2 = np.random.rand(3, self.kernel.ndim)
        hyper = self.kernel.get_hyper()
        K1 = self.kernel.get(x1, x2)
        K2 = [self.kfun(hyper, x1[i], x2[j]) for (i,j) in np.ndindex(*K1.shape)]
        K2 = np.reshape(K2, K1.shape)
        nt.assert_allclose(K1, K2)

    def test_grad(self):
        x1 = np.random.rand(5, self.kernel.ndim)
        x2 = np.random.rand(3, self.kernel.ndim)
        hyper = self.kernel.get_hyper()
        G1 = np.array(list(self.kernel.grad(x1, x2)))
        G2 = [self.dhfun(hyper, x1[i], x2[j]) for (i,j) in np.ndindex(*G1[0].shape)]
        G2 = np.reshape(np.array(G2).T, G1.shape)
        nt.assert_allclose(G1, G2)

    def test_dget(self):
        x1 = np.random.rand(5, self.kernel.ndim)
        k1 = self.kernel.dget(x1)
        k2 = np.diag(self.kernel.get(x1))
        nt.assert_allclose(k1, k2)

    def test_dgrad(self):
        x1 = np.random.rand(5, self.kernel.ndim)
        g1 = list(self.kernel.dgrad(x1))
        g2 = map(np.diag, list(self.kernel.grad(x1)))
        nt.assert_allclose(g1, g2)


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
# tests for the SE kernels.

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
