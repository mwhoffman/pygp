import os
import numpy as np

import pygp.likelihoods as pgl
import pygp.kernels as pgk
import pygp.inference as pgi


if __name__ == '__main__':
    sn  = 0.14556133488
    ell = 0.307661277438
    sf  = 1.13604778127

    # create our GP model.
    likelihood = pgl.Gaussian(sn)
    kernel = pgk.SEARD(ell, sf)
    gp = pgi.ExactGP(likelihood, kernel)

    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # add data to the model.
    gp.add_data(X, y)
