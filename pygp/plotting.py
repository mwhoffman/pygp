"""
Plotting methods for GP objects.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import matplotlib.pyplot as pl

# local imports
from .utils.models import get_params

# exported symbols
__all__ = ['gpplot', 'sampleplot']


def gpplot(gp, xmin=None, xmax=None, nsamples=None, mean=True, data=True,
               error=True, delta=0.05, spaghetti=False,
               figure=None, clear=True, draw=True):
    xmin = gp._X[:,0].min() if (xmin is None) else xmin
    xmax = gp._X[:,0].max() if (xmax is None) else xmax

    x = np.linspace(xmin, xmax, 500)
    mu, lo, hi = gp.predict(x[:,None], delta=delta)

    # get the current/named figure and clear it if requested.
    fg = pl.gcf() if (figure is None) else pl.figure(figure)
    if clear: fg.clf()

    # get the axes and clear that.
    ax = fg.gca()
    ax.cla()

    if error:
        ax.fill_between(x, lo, hi, color='k', alpha=0.1, zorder=1)

    if nsamples:
        f = gp.sample(x[:,None], n=nsamples)
        if spaghetti:
            ax.plot(x, f.T, color='m', alpha=0.1, zorder=1)
        else:
            ax.plot(x, f.T, lw=2, alpha=0.75)

    if mean:
        ax.plot(x, mu, color='k', zorder=2, lw=2)

    if data and gp._X is not None:
        ax.scatter(gp._X.ravel(), gp._y, color='b', s=30, zorder=3)

    ax.axis('tight')
    ax.axis(xmin=xmin, xmax=xmax)

    if draw:
        ax.figure.canvas.draw()


def sampleplot(gp, samples, figure=None, draw=True):
    samples = samples.copy()
    naxes = samples.shape[1]
    labels = []

    for name, block, log in get_params(gp):
        size = block.stop - block.start
        if size == 1:
            labels.append(name)
        else:
            labels.extend('%s[%d]' % (name, i) for i in xrange(size))
        if log:
            np.exp(samples[:, block], out=samples[:, block])

    fg = pl.gcf()
    fg.clf()

    # get the current/named figure and clear it.
    fg = pl.gcf() if (figure is None) else pl.figure(figure)
    fg.clf()

    for i, j in np.ndindex(naxes, naxes):
        if i >= j:
            continue
        ax = fg.add_subplot(naxes-1, naxes-1, (j-1)*(naxes-1)+i+1)
        ax.scatter(samples[:,i], samples[:,j], alpha=0.1)

        if i == 0:
            ax.set_ylabel(labels[j])
        else:
            ax.set_yticklabels([])

        if j == naxes-1:
            ax.set_xlabel(labels[i])
        else:
            ax.set_xticklabels([])

    if draw:
        fg.canvas.draw()

