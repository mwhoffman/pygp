import os
import numpy as np
import pygp as pg


if __name__ == '__main__':
    # load the data.
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data to it.
    gp = pg.BasicGP(.1, .1, 1)
    gp.add_data(X, y)

    pg.optimize(gp)
    pg.gpplot(gp)
