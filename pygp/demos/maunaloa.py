"""
Simple demo showing how to use a more complicated kernel. This fits the data of
CO_2 levels at Mauna Loa; see chapter 5 of Rasmussen and Williams. Note that we
don't fit the hyperparameters as the values given below are reasonable and this
would just take some time.
"""

import os
import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pygp.kernels as pk


if __name__ == '__main__':
    # load the file from the current directory and get rid of any censored data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.loadtxt(os.path.join(cdir, 'maunaloa.txt')).flatten()
    data = np.array([(x, y) for x, y in enumerate(data) if y > -99])

    # minor manipulations of the data to make the ranges reasonable.
    X = data[:, 0, None] / 12. + 1958
    y = data[:, 1]

    # these are near the values called for in Rasmussen and Williams, so they
    # should give reasonable results and thus we'll skip the fit.
    kernel = \
        pk.SE(67, 66) + \
        pk.SE(2.4, 90) * pk.Periodic(1, 1, 1) + \
        pk.RQ(1.2, .66, 0.78) + \
        pk.SE(0.15, 0.15)

    # use a gaussian likeihood with this standard deviation.
    likelihood = pygp.likelihoods.Gaussian(sigma=0.2)

    # construct the model and add the data.
    gp = pygp.inference.ExactGP(likelihood, kernel, y.mean())
    gp.add_data(X, y)

    # plot everything.
    pl.figure(1)
    pl.clf()
    pp.plot_posterior(gp, mean=False, xmax=2020, marker='.')
    pl.legend(loc='upper left')
    pl.draw()
    pl.show()
