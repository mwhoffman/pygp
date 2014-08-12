"""
Basic demo showing how to instantiate a simple GP model, add data to it, and
optimize its hyperparameters.
"""

# global imports.
import os
import numpy as np

# local imports
import pygp
import pygp.plotting


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp.add_data(X, y)

    # find the ML parameters and sample from the posterior.
    pygp.optimize(gp)
    pygp.plotting.plot(gp, figure=1)
