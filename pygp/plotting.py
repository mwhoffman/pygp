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

# exported symbols
__all__ = ['plot_posterior', 'plot_samples']


def plot_posterior(model,
                   xmin=None, xmax=None,
                   mean=True, data=True, error=True, pseudoinputs=True,
                   lw=2, ls='-', color=None, marker='o', marker2='x'):
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
        raise ValueError('bounds must be given if no data is present')

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

    if color is None:
        color = next(ax._get_lines.color_cycle)

    # default arguments for markers.
    margs = dict(color='k', zorder=3)
    margs = {
        'x': dict(marker='x', facecolors='none', s=30, lw=1, **margs),
        'o': dict(marker='o', facecolors='none', s=30, lw=1, **margs),
        '*': dict(marker='*', facecolors='none', s=30, lw=1, **margs),
        ',': dict(marker=',', facecolors='none', s=30, lw=1, **margs),
        '.': dict(marker='.', **margs)}

    if mean:
        # plot the mean
        ax.plot(x, mu, lw=lw, ls=ls, label='mean', color=color)

    if error:
        # plot the error bars and add an empty plot that will be used by the
        # legend if it's called for.
        alpha = 0.25
        ax.fill_between(x, lo, hi, color=color, alpha=alpha)
        ax.plot([], [], color=color, alpha=alpha, linewidth=10,
                label='uncertainty')

    if data and X is not None:
        # plot the data; use smaller markers if we have a lot of data.
        ax.scatter(X.ravel(), y, label='data', **margs[marker])

    if hasattr(model, 'pseudoinputs') and pseudoinputs:
        # plot any pseudo-inputs.
        ymin, ymax = ax.get_ylim()
        u = model.pseudoinputs.ravel()
        v = np.full_like(u, ymin + 0.1 * (ymax-ymin))
        ax.scatter(u, v, label='pseudo-inputs', **margs[marker2])

    pl.axis('tight')


def plot_samples(model):
    """
    Plot the posterior over hyperparameters for a sample-based meta model.
    """
    # get the figure and clear it.
    fg = pl.gcf()
    fg.clf()

    samples = np.array(list(m.get_hyper() for m in model))
    values = np.zeros((samples.shape[0], 0))
    labels = []

    offset = 0
    for key, size, log in next(model.__iter__())._params():
        for i in xrange(size):
            vals = samples[:, offset+i]
            name = key + ('' if (size == 1) else '_%d' % i)
            if not np.allclose(vals, vals[0]):
                values = np.c_[values, np.exp(vals) if log else vals]
                labels.append(name)
        offset += size

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
