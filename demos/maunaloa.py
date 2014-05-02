"""
Simple demo showing how to use a more complicated kernel. This fits the data of
CO_2 levels at Mauna Loa; see chapter 5 of Rasmussen and Williams. Note that we
don't fit the hyperparameters as the values given below are reasonable and this
would just take some time.
"""

import os
import numpy as np

import pygp as pg
import pygp.kernels as pk


# load the file from the current directory and get rid of any censored data.
cdir = os.path.abspath(os.path.dirname(__file__))
data = np.loadtxt(os.path.join(cdir, 'maunaloa.txt')).flatten()
data = np.array([(x,y) for (x,y) in zip(np.arange(len(data)), data) if y>-99])

X = data[:,0,None] / 12.
y = data[:,1]
y -= y.mean()

# use the empirical mean as our prior.
sigma = 0.1

# these are near the values called for in Rasmussen and Williams, so they should
# give reasonable results and thus we'll skip the fit.
kernel = pk.SEIso(60, 60) + \
         pk.SEIso(90, 2) * pk.Periodic(1, 1, 1) + \
         pk.RQIso(0.7, 1.2, 0.7) + \
         pk.SEIso(0.15, 0.15)

likelihood = pg.likelihoods.Gaussian(sigma)
gp = pg.ExactGP(likelihood, kernel)

gp.add_data(X, y)
