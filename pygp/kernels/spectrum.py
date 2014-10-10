"""
Implementation of the spectrum kernel.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import Kernel

from ..utils.models import printable

# exported symbols
__all__ = ['Spectrum']


@printable
class Spectrum(Kernel):
    def __init__(self, weights=None, k=1, alphabet="ACGT"):
        self._alphabet = alphabet
        self._k = k
        self.nhyper = len(alphabet)**k

        if weights==None:
            self._log_weights = np.zeros(self.nhyper, dtype=float)
        else:
            if self.nhyper!=len(weights):
                raise ValueError('size of weights must equal alphabet size to the power of k')
            self._log_weights = np.log(np.array(weights, ndmin=1, dtype=float))

    def transform(self, X):
        t = [calculate_ngram_frequency(seq[0], self._k, self._alphabet) for seq in X]
        return np.array(t)

    def _params(self):
        return [
            ('weights', len(self._log_weights)),
        ]

    def get_hyper(self):
        return np.r_[self._log_weights]

    def set_hyper(self, hyper):
        self._log_weights = np.array(hyper, ndmin=1, dtype=float)

    def get(self, X1, X2=None):
        if X2==None:
            X2 = X1

        W = np.diag(np.exp(self._log_weights))
        K = np.zeros((X1.shape[0],X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i,j] = np.dot(np.dot(X1[i], W), X2[j])

        return K

    def dget(self, X1):
        W = np.diag(np.exp(self._log_weights))
        D = np.zeros(X1.shape[0])
        for i in range(X1.shape[0]):
            D[i] = np.dot(np.dot(X1[i], W), X1[i])
        return D

    def grad(self, X1, X2=None):
        if X2==None:
            X2 = X1
        
        for p in range(len(self._log_weights)):
            G = np.zeros((X1.shape[0],X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    G[i,j] = X1[i,p] * X2[j,p] * np.exp(self._log_weights[p])
            yield G

    def dgrad(self, X1):
        for p in range(len(self._log_weights)):
            G = np.zeros(X1.shape[0])
            for i in range(X1.shape[0]):
                G[i] = X1[i]**2
            yield G

def calculate_ngram_frequency(seq, n, alphabet):
    ngram_frequency = np.zeros(len(alphabet)**n, dtype=float)

    for i in range(len(seq)-n+1):
        ngram = seq[i:i+n]
        index = ngram_to_index(ngram, alphabet)
        ngram_frequency[index] += 1

    return ngram_frequency

def ngram_to_index(ngram, alphabet):
    index = 0
    
    for i,b in enumerate(ngram):
        index += alphabet.index(b) * len(alphabet)**i

    return index
