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

    # create a basic GP.
    gp1 = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp1.add_data(X, y)

    # create a sparse GPs.
    U = np.linspace(-1.3, 2, 10)[:, None]
    gp2 = pygp.inference.FITC.from_gp(gp1, U)
    gp3 = pygp.inference.DTC.from_gp(gp1, U)

    # find the ML parameters
    pygp.optimize(gp1)
    pygp.optimize(gp2)
    pygp.optimize(gp3)

    # plot the dense gp.
    pl.figure(1)
    pl.clf()
    pl.subplot(131)
    pp.plot_posterior(gp1)
    pl.title('Full GP')

    # grab the axis limits.
    axis = pl.axis()

    # plot the FITC sparse gp.
    pl.subplot(132)
    pp.plot_posterior(gp2, pseudoinputs=True)
    pl.title('Sparse GP (FITC)')
    pl.axis(axis)
    pl.draw()

    # plot the sparse gp.
    pl.subplot(133)
    pp.plot_posterior(gp3, pseudoinputs=True)
    pl.title('Sparse GP (DTC)')
    pl.axis(axis)
    pl.legend(loc='upper left')
    pl.draw()
    pl.show()
