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


def gpplot(gp,
           xmin=None, xmax=None, mean=True, data=True, error=True, delta=0.05,
           xlabel='', ylabel='', title='', figure=None, subplot=None, draw=True):

    # get the axes object and clear it.
    fg = pl.gcf() if (figure is None) else pl.figure(figure)
    if subplot is None:
        fg.clf()
        ax = fg.gca()
    else:
        ax = fg.add_subplot(subplot)
        ax.cla()

    xmin = gp._X[:,0].min() if (xmin is None) else xmin
    xmax = gp._X[:,0].max() if (xmax is None) else xmax

    x = np.linspace(xmin, xmax, 500)
    mu, lo, hi = gp.predict(x[:,None], delta=delta)

    if error:
        ax.fill_between(x, lo, hi, color='k', alpha=0.1, zorder=1)

    if mean:
        ax.plot(x, mu, color='k', zorder=2, lw=2)

    if data and gp._X is not None:
        ax.scatter(gp._X.ravel(), gp._y, color='b', s=30, zorder=3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('tight')
    ax.axis(xmin=xmin, xmax=xmax)

    if draw:
        ax.figure.canvas.draw()


def sampleplot(model, samples,
               figure=None, draw=True):

    # get the figure and clear it.
    fg = pl.gcf() if (figure is None) else pl.figure(figure)
    fg.clf()

    values = np.zeros((samples.shape[0], 0))
    labels = []

    for key, block, log in get_params(model):
        for i in range(block.start, block.stop):
            vals = samples[:,i]
            size = block.stop - block.start
            name = key + ('' if (size == 1) else '_%d' % (i - block.start))
            if not np.allclose(vals, vals[0]):
                values = np.c_[values, np.exp(vals) if log else vals]
                labels.append(name)

    naxes = values.shape[1]

    if naxes == 1:
        ax = fg.add_subplot(111)
        ax.hist(values[:,0], bins=20)
        ax.set_xlabel(labels[0])
        ax.set_yticklabels([])

    else:
        for i, j in np.ndindex(naxes, naxes):
            if i >= j: continue
            ax = fg.add_subplot(naxes-1, naxes-1, (j-1)*(naxes-1)+i+1)
            ax.scatter(values[:,i], values[:,j], alpha=0.1)

            if i == 0: ax.set_ylabel(labels[j])
            else: ax.set_yticklabels([])

            if j == naxes-1: ax.set_xlabel(labels[i])
            else: ax.set_xticklabels([])

    if draw:
        fg.canvas.draw()

