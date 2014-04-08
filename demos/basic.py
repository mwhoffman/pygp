import os
import numpy as np
import pygp.inference as pgi
import pygp.plotting as pgp


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # parameters we'll use.
    sn  = 0.14556133488
    ell = 0.307661277438
    sf  = 1.13604778127

    # create the model, add data, and plot it.
    gp = pgi.BasicGP(sn, ell, sf)
    gp.add_data(X, y)
    pgp.gpplot(gp)
