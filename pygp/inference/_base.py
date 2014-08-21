"""
Interface for latent function inference in Gaussian process models. These
models will assume that the hyperparameters are fixed and any optimization
and/or sampling of these parameters will be left to a higher-level wrapper.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.linalg as sla

# local imports
from ..utils.abc import abstractmethod
from ..utils.models import Parameterized, dot_params
from ..utils.random import rstate
from ._fourier import FourierSample

# exported symbols
__all__ = ['GP']


class GP(Parameterized):
    """
    GP inference interface.

    This class defines the GP interface. Although it implements the
    Parameterized interface it defines additional abstract methods and is still
    abstract as a result.

    The methods that must be implemented are:

        `_update`: update internal statistics given all the data.
        `_posterior`: compute the full posterior for use with sampling.
        `posterior`: compute the marginal posterior and its gradient.
        `loglikelihood`: compute the loglikelihood of observed data.

    Additionally, the following method can be implemented for improved
    performance in some circumstances:

        `_updateinc`: incremental update given new data.
    """
    def __init__(self, likelihood, kernel):
        self._likelihood = likelihood
        self._kernel = kernel
        self._X = None
        self._y = None

        self.nhyper = (self._likelihood.nhyper +
                       self._kernel.nhyper)

    def __repr__(self):
        models = [repr(self._likelihood),
                  repr(self._kernel)]

        string = self.__class__.__name__ + '('
        joiner = ',\n' + (len(string) * ' ')
        string += joiner.join(models) + ')'

        return string

    def _params(self):
        params = dot_params('like', self._likelihood._params())
        params += dot_params('kern', self._kernel._params())
        return params

    def get_hyper(self):
        # NOTE: if subclasses define any "inference" hyperparameters they can
        # implement their own get/set methods and call super().
        return np.r_[self._likelihood.get_hyper(),
                     self._kernel.get_hyper()]

    def set_hyper(self, hyper):
        a = 0
        for model in [self._likelihood, self._kernel]:
            model.set_hyper(hyper[a:a+model.nhyper])
            a += model.nhyper
        if self.ndata > 0:
            self._update()

    @property
    def ndata(self):
        """The number of current input/output data pairs."""
        return 0 if (self._X is None) else self._X.shape[0]

    @property
    def data(self):
        """The current input/output data."""
        return (self._X, self._y)

    def add_data(self, X, y):
        """
        Add new data to the GP model.
        """
        X = self._kernel.transform(X)
        y = self._likelihood.transform(y)

        if self._X is None:
            self._X = X.copy()
            self._y = y.copy()
            self._update()

        else:
            try:
                self._updateinc(X, y)
                self._X = np.r_[self._X, X]
                self._y = np.r_[self._y, y]

            except NotImplementedError:
                self._X = np.r_[self._X, X]
                self._y = np.r_[self._y, y]
                self._update()

    def sample(self, X, m=None, latent=True, rng=None):
        """
        Sample values from the posterior at points `X`. Given an `(n,d)`-array
        `X` this will return an `n`-vector corresponding to the resulting
        sample.

        If `m` is not `None` an `(m,n)`-array will be returned instead,
        corresponding to `m` such samples. If `latent` is `False` the sample
        will instead be returned corrupted by the observation noise. Finally
        `rng` can be used to seed the randomness.
        """
        X = self._kernel.transform(X)
        flatten = (m is None)
        m = 1 if flatten else m
        n = len(X)

        # if a seed or instantiated RandomState is given use that, otherwise
        # use the global object.
        rng = rstate(rng)

        # add a tiny amount to the diagonal to make the cholesky of Sigma
        # stable and then add this correlated noise onto mu to get the sample.
        mu, Sigma = self._posterior(X)
        Sigma += 1e-10 * np.eye(n)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), sla.cholesky(Sigma))

        if not latent:
            f = self._likelihood.sample(f.ravel(), rng).reshape(m, n)

        return f.ravel() if flatten else f

    def sample_fourier(self, N, rng=None):
        """
        Approximately sample a function from the GP using a fourier-basis
        expansion with N bases. See the documentation on `FourierSample` for
        details on the returned function object.
        """
        return FourierSample(N, self._likelihood, self._kernel, self._X,
                             self._y, rng)

    @abstractmethod
    def _update(self):
        """
        Update any internal parameters (ie sufficient statistics) given the
        entire set of current data.
        """

    # NOTE: the following method is not abstract since we don't require that it
    # is implemented. if it is not implemented the full _update is performed
    # when new data is added.

    def _updateinc(self, X, y):
        """
        Update any internal parameters given additional data in the form of
        input/output pairs `X` and `y`. This method is called before data is
        appended to the internal data-store and no subsequent call to `_update`
        is performed.
        """
        raise NotImplementedError

    @abstractmethod
    def _posterior(self, X):
        """
        Compute the posterior at points `X`. This should return the mean and
        full covariance matrix of the given points.
        """

    @abstractmethod
    def posterior(self, X, grad=False):
        """
        Compute the marginal posterior at points `X`. This should return the
        mean and variance of the given points, and if `grad == True` should
        return their derivatives with respect to the input location as well
        (i.e. a 4-tuple).
        """

    @abstractmethod
    def loglikelihood(self, grad=False):
        """
        Return the marginal loglikelihood of the data. If `grad == True` also
        return the gradient with respect to the hyperparameters.
        """
