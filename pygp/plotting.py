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
__all__ = ['gpplot']


def gpplot(gp, xmin=None, xmax=None, nsamples=None, mean=True, data=True,
               error=True, ci=0.95, spaghetti=False):
    xmin = xmin or gp._X[:,0].min()
    xmax = xmax or gp._X[:,0].max()

    x = np.linspace(xmin, xmax, 500)
    mu, lo, hi = gp.predict(x[:,None], ci=ci)

    ax = pl.gca()
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

    if data:
        ax.scatter(gp._X.ravel(), gp._y, color='b', s=30, zorder=3)

    ax.axis('tight')
    ax.axis(xmin=xmin, xmax=xmax)
    ax.figure.canvas.draw()
