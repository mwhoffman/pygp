"""
Implementation of the squared-exponential kernels.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import RealKernel
from ._distances import rescale, diff, sqdist, sqdist_foreach

from ..utils.random import rstate
from ..utils.models import printable

# exported symbols
__all__ = ['SE']


@printable
class SE(RealKernel):
    def __init__(self, sf, ell, ndim=None):
        self._logsf = np.log(float(sf))
        self._logell = np.log(ell)
        self._iso = False
        self.ndim = np.size(self._logell)
        self.nhyper = 1 + np.size(self._logell)

        if ndim is not None:
            if np.size(self._logell) == 1:
                self._logell = float(self._logell)
                self._iso = True
                self.ndim = ndim
            else:
                raise ValueError('ndim only usable with scalar lengthscales')

    def _params(self):
        return [
            ('sf', 1),
            ('ell', self.nhyper-1),
        ]

    def get_hyper(self):
        return np.r_[self._logsf, self._logell]

    def set_hyper(self, hyper):
        self._logsf = hyper[0]
        self._logell = hyper[1] if self._iso else hyper[1:]

    def get(self, X1, X2=None):
        X1, X2 = rescale(np.exp(self._logell), X1, X2)
        return np.exp(self._logsf*2 - sqdist(X1, X2)/2)

    def grad(self, X1, X2=None):
        X1, X2 = rescale(np.exp(self._logell), X1, X2)
        D = sqdist(X1, X2)
        K = np.exp(self._logsf*2 - D/2)
        yield 2*K                               # derivative wrt logsf
        if self._iso:
            yield K*D                           # derivative wrt logell (iso)
        else:
            for D in sqdist_foreach(X1, X2):
                yield K*D                       # derivatives wrt logell (ard)

    def dget(self, X1):
        return np.exp(self._logsf*2) * np.ones(len(X1))

    def dgrad(self, X):
        yield 2 * self.dget(X)
        for _ in xrange(self.nhyper-1):
            yield np.zeros(len(X))

    def gradx(self, X1, X2=None):
        ell = np.exp(self._logell)
        X1, X2 = rescale(ell, X1, X2)

        D = diff(X1, X2)
        K = np.exp(self._logsf*2 - np.sum(D**2, axis=-1)/2)
        G = -K[:, :, None] * D / ell
        return G

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        sf2 = np.exp(self._logsf*2)
        ell = np.exp(self._logell)
        W = rng.randn(N, self.ndim) / ell
        return W, sf2

    # def get_local(self, x, X1, X2=None):
    #     # FIXME! doesn't work for ard kernels.
    #     ell = np.exp(self._logell*-2)
    #     sf2 = np.exp(self._logsf*2)
    #     return local_se(ell, sf2, 0.0, x, X1, X2)

    def gradxy(self, X1, X2=None):
        """
        Derivatives of the kernel with respect to both its first and second
        arguments. This corresponds to the covariance between gradients values
        evaluated at X1 and at X2. Returns an (m,n,d,d)-array.
        """
        ell = np.exp(self._logell)
        X1, X2 = rescale(ell, X1, X2)
        D = diff(X1, X2)
        m, n, d = D.shape

        K = np.exp(self._logsf*2 - np.sum(D**2, axis=-1)/2)
        D /= ell
        M = np.eye(d)/ell**2 - D[:,:,None] * D[:,:,:,None]
        G = M*K[:,:,None,None]

        return G

    def gradyxx(self, X1, X2=None):
        ell = np.exp(self._logell)
        lam = np.ones(d) / ell**2
        X1, X2 = rescale(ell, X1, X2)
        D = diff(X1, X2)
        m, n, d = D.shape

        # the K and M computed above. here we have the negative of the earlier M
        # though. the first two indices correspond to the components of X1/X2.
        K = np.exp(self._logsf*2 - np.sum(D**2, axis=-1)/2)
        D /= ell
        M = D[:,:,None] * D[:,:,:,None] - np.diag(lam)

        # let i be the 3rd index which corresponds to the portion of y (i.e. x2)
        # we're differentiating with respect to. Let v be the corresponding
        # vector in D[p,q]

        # computes [lambda_i (e_i v' + v e_i')] for each block.
        A = np.reshape(np.eye(d).flatten()[None,None,:,None] * D[:,:,None,:], (m,n,d,d,d))
        A += A.swapaxes(-1,-2)
        A *= lam[None,None,:,None,None]

        # computes [v_i M - A_i] for each block.
        W = M[:,:,None] * D[:,:,:,None,None] - A
        G = K[:,:,None,None,None] * W

        return G
