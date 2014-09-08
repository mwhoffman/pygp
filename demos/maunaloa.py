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


# load the file from the current directory and get rid of any censored data.
cdir = os.path.abspath(os.path.dirname(__file__))
data = np.loadtxt(os.path.join(cdir, 'maunaloa.txt')).flatten()
data = np.array([(x, y) for x, y in zip(np.arange(len(data)), data) if y>-99])

# minor manipulations of the data to make the ranges reasonable. also use the
# empirical mean as the prior mean.
X = data[:, 0, None] / 12.
y = data[:, 1]
y -= y.mean()

# these are near the values called for in Rasmussen and Williams, so they
# should give reasonable results and thus we'll skip the fit.
kernel = \
    pk.SE(60, 60) + \
    pk.SE(2, 90) * pk.Periodic(1, 1, 1) + \
    pk.RQ(0.7, 1.2, 0.7) + \
    pk.SE(0.15, 0.15)

# use a gaussian likeihood with this standard deviation.
likelihood = pygp.likelihoods.Gaussian(sigma=0.1)

# construct the model and add the data.
gp = pygp.inference.ExactGP(likelihood, kernel)
gp.add_data(X, y)

# plot everything.
pl.figure(1)
pl.clf()
pp.plot_posterior(gp, mean=False, xmax=70)
pl.legend(loc='upper left')
pl.draw()
pl.show()
