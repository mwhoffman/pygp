"""
Learning tests.
"""

# pylint: disable=no-member
# pylint: disable=missing-docstring

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import numpy.testing as nt

import pygp
import pygp.demos.basic as demo


def test_optimization():
    # load the data.
    cdir = os.path.abspath(os.path.dirname(demo.__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))
    X = data['X']
    y = data['y']

    # create the model and add data.
    gp = pygp.BasicGP(sn=.1, sf=1, ell=.1, mu=0)
    gp.add_data(X, y)

    # optimize the model
    pygp.optimize(gp, {'sn': None})

    # make sure our constraint is satisfied
    nt.assert_equal(gp.get_hyper()[0], np.log(0.1))
