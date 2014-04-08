import os
import numpy as np
import matplotlib.pyplot as pl

import pygp.likelihoods as pgl
import pygp.kernels as pgk
import pygp.inference as pgi


if __name__ == '__main__':
    sn  = 0.14556133488
    ell = 0.307661277438
    sf  = 1.13604778127

    # create our GP model.
    gp = pgi.BasicGP(sn, ell, sf)

    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # add data to the model.
    gp.add_data(X, y)

    # for some test points get the mean and 95% credible interval.
    xmin = -1.5
    xmax = 2.2
    xt = np.linspace(xmin, xmax, 500)
    ft = gp.sample(xt[:,None], n=3)
    mu, lo, hi = gp.predict(xt[:,None], ci=0.95)

    # plot it.
    pl.figure(1)
    pl.clf()
    pl.fill_between(xt, lo, hi, color='k', alpha=0.1)
    pl.plot(xt, mu, color='k')
    pl.plot(xt, ft.T)
    pl.axis('tight')
    pl.axis(xmin=xmin, xmax=xmax)
    pl.scatter(X.ravel(), y, color='b', s=20)
    pl.draw()
