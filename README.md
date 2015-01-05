pygp
----

A Python package for inference with Gaussian processes.

The goal is for this to be a relatively self-contained python package for using
Gaussian Processes (GPs) that is loosely based on Carl Rasmussen's `GPML`
toolbox. The structure and ideas are based on that toolbox's implementations,
but some changes have been made to make this package more pythonic.

[![Build Status](https://travis-ci.org/mwhoffman/pygp.svg)]
(https://travis-ci.org/mwhoffman/pygp)
[![Coverage Status](https://coveralls.io/repos/mwhoffman/pygp/badge.png)]
(https://coveralls.io/r/mwhoffman/pygp)

Installation
============

The easiest way to install this package is by running

    pip install -r https://github.com/mwhoffman/pygp/raw/master/requirements.txt
    pip install git+https://github.com/mwhoffman/pygp.git

The first line installs any dependencies of the package and the second line
installs the package itself. Alternatively the repository can be cloned directly
in order to make any local modifications to the code. In this case the
dependencies can easily be installed by running

    pip install -r requirements.txt

from the main directory. The package itself can be installed by running `python
setup.py` or by symlinking the directory into somewhere on the `PYTHONPATH`.
Once the package is installed the included demos can be run directly via python.
For example, by running

    python -m pygp.demos.basic

A full list of demos can be viewed [here](pygp/demos).
