"""
Basic demo showing how to instantiate a simple GP model, add data to it, and
optimize its hyperparameters.
"""

# global imports.
import os
import numpy as np
import matplotlib.pyplot as pl

# local imports
import pygp
import pygp.plotting as pp


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model, add data, and optimize it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)
    pygp.optimize(gp)

    # plot the posterior.
    pl.figure(1)
    pl.clf()
    pp.plot_posterior(gp)
    pl.legend(loc='best')
    pl.draw()
    pl.show()
