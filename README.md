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

    pip install git+https://github.com/mwhoffman/{mwhutils,pygp}.git

Alternatively the packages above can be installed by cloning their repositories
and using `setup.py` directly. Once the package is installed the included demos
can be run directly via python.  For example, by running

    python -m pygp.demos.basic

A full list of demos can be viewed [here](pygp/demos).
