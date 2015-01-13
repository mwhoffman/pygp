"""
Tests for the plotting convenience functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import numpy.testing as nt
import mwhutils.random as mr

import pygp
import pygp.plotting as pg


def test_plot_posterior():
    gp = pygp.BasicGP(1, 1, 1)
    gp = pygp.inference.FITC.from_gp(gp, mr.grid((0, 1), 10))

    nt.assert_raises(ValueError, pg.plot_posterior, gp)

    pg.plot_posterior(gp, 0, 1)
    pg.plot_posterior(gp, 0, 1, error=False)
    pg.plot_posterior(gp, 0, 1, mean=False)
    pg.plot_posterior(gp, 0, 1, data=False)
    pg.plot_posterior(gp, 0, 1, pseudoinputs=False)
    pg.plot_posterior(gp, 0, 1, color='b')
    pg.plot_posterior(gp, 0, 1, lw=3)
    pg.plot_posterior(gp, 0, 1, ls='.')

    X = mr.grid((0, 1), 10)
    y = np.random.rand(10)

    gp.add_data(X, y)
    pg.plot_posterior(gp)


def test_plot_samples():
    # create a gp model
    model = pygp.BasicGP(sn=1, sf=1, ell=1)

    # create a prior datastructure
    priors1 = {'sn': pygp.priors.Uniform(0.01, 1.0),
               'sf': pygp.priors.Uniform(0.01, 5.0),
               'ell': pygp.priors.Uniform(0.01, 1.0),
               'mu': pygp.priors.Uniform(-2, 2)}

    # create a prior datastructure which holds fixed all but one parameter
    priors2 = {'sn': pygp.priors.Uniform(0.01, 1.0),
               'sf': None,
               'ell': None,
               'mu': None}

    # sample and plot the models
    pg.plot_samples(pygp.meta.SMC(model, priors1, 200))
    pg.plot_samples(pygp.meta.SMC(model, priors2, 200))
