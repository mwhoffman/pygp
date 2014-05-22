"""
Basic demo showing how to instantiate a simple GP model, add data to it, and
optimize its hyperparameters.
"""

# global imports.
import os
import numpy as np
import matplotlib.pyplot as pl
import mpl_toolkits.mplot3d

# local imports
import pygp
import pygp.hyper.priors as pgp


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)

    pygp.optimize(gp)

    priors = dict(
        sn =None,
        sf =pgp.Uniform(0.01, 4.0),
        ell=pgp.Uniform(0.01, 1.0))

    hyper = pygp.hyper.sample(gp, priors, 10000)

    pl.figure(1)
    pygp.gpplot(gp)

    fg = pl.figure(2)
    ax = pl.gca()
    ax.cla()
    ax.scatter(np.exp(hyper[:,1]),
               np.exp(hyper[:,2]), alpha=0.1)
    ax.set_xlabel('sf')
    ax.set_ylabel('ell')
    ax.figure.canvas.draw()
