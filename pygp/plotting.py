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
__all__ = ['plot_posterior', 'sampleplot']


def plot_posterior(model,
                   xmin=None, xmax=None,
                   mean=True, data=True, error=True, pseudoinputs=False):
    """
    Plot a one-dimensional posterior model.

    Parameters:
        xmin: minimum x value
        xmax: maximum x value
        mean: plot the mean
        data: plot the data
        error: plot the error bands
        pseudoinputs: plot pseudoinputs (if there are any)
    """

    # grab the data.
    X, y = model.data
    if X is None and (xmin is None or xmax is None):
        raise Exception('bounds must be given if no data is present')

    # get the input points.
    xmin = X[:, 0].min() if (xmin is None) else xmin
    xmax = X[:, 0].max() if (xmax is None) else xmax
    x = np.linspace(xmin, xmax, 500)

    # get the mean and confidence bands.
    mu, s2 = model.posterior(x[:, None])
    lo = mu - 2 * np.sqrt(s2)
    hi = mu + 2 * np.sqrt(s2)

    # get the axes.
    ax = pl.gca()

    if mean:
        # plot the mean
        ax.plot(x, mu, lw=2, color='b', label='mean')

    if error:
        # plot the error bars and add an empty plot that will be used by the
        # legend if it's called for.
        ax.fill_between(x, lo, hi, color='k', alpha=0.15)
        ax.plot([], [], color='k', alpha=0.15, linewidth=10,
                label='uncertainty')

    if data and X is not None:
        # plot the data; use smaller markers if we have a lot of data.
        if len(X) > 100:
            ax.scatter(X.ravel(), y, marker='.', color='k', label='data')
        else:
            ax.scatter(X.ravel(), y, s=20, lw=1, facecolors='none', color='k',
                       label='data')

    if hasattr(model, 'pseudoinputs') and pseudoinputs:
        # plot any pseudo-inputs.
        ymin, ymax = ax.get_ylim()
        U = model.pseudoinputs.ravel()
        ax.scatter(U, np.ones_like(U) * (ymin + 0.1 * (ymax-ymin)),
                   s=20, lw=1, marker='x', color='k', label='pseudo-inputs')


def sampleplot(model, samples,
               figure=None, draw=True):

    # get the figure and clear it.
    fg = pl.gcf() if (figure is None) else pl.figure(figure)
    fg.clf()

    values = np.zeros((samples.shape[0], 0))
    labels = []

    for key, block, log in get_params(model):
        for i in range(block.start, block.stop):
            vals = samples[:, i]
            size = block.stop - block.start
            name = key + ('' if (size == 1) else '_%d' % (i - block.start))
            if not np.allclose(vals, vals[0]):
                values = np.c_[values, np.exp(vals) if log else vals]
                labels.append(name)

    naxes = values.shape[1]

    if naxes == 1:
        ax = fg.add_subplot(111)
        ax.hist(values[:, 0], bins=20)
        ax.set_xlabel(labels[0])
        ax.set_yticklabels([])

    else:
        for i, j in np.ndindex(naxes, naxes):
            if i >= j:
                continue
            ax = fg.add_subplot(naxes-1, naxes-1, (j-1)*(naxes-1)+i+1)
            ax.scatter(values[:, i], values[:, j], alpha=0.1)

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
