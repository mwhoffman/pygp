"""
Basic demo plotting the resulting hyperparameter samples by using MCMC.
"""

# global imports.
import os
import numpy as np
import matplotlib.pyplot as pl

# local imports
import pygp
import pygp.priors
import pygp.plotting as pp


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    model = pygp.BasicGP(sn=.1, sf=1, ell=.1)
    model.add_data(X, y)

    # find the ML hyperparameters and plot the predictions.
    pygp.optimize(model)

    # create a prior structure.
    priors = dict(
        sn=pygp.priors.Uniform(0.01, 1.0),
        sf=pygp.priors.Uniform(0.01, 5.0),
        ell=pygp.priors.Uniform(0.01, 1.0))

    # create a sample-based model.
    mcmc = pygp.meta.MCMC(model, priors, n=5000)

    pl.figure(1)
    pp.plot_samples(mcmc)
    pl.draw()
    pl.show()
