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

    # create a basic GP.
    gp1 = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    gp1.add_data(X, y)

    # create a sparse GP.
    U = np.linspace(-1.3, 2, 10)[:, None]
    gp2 = pygp.inference.FITC.from_gp(gp1, U)

    # find the ML parameters for both
    pygp.optimize(gp1)
    pygp.optimize(gp2)

    # plot them.
    pygp.plotting.plot(gp1, figure=1, subplot=121, ymin=-2.5, ymax=3,
                       title='Full GP')

    pygp.plotting.plot(gp2, figure=1, subplot=122, ymin=-2.5, ymax=3,
                       title='Sparse GP', pseudoinputs=True, legend=True)
