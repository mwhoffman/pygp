"""
Setup script for pygp.
"""

from setuptools import setup, find_packages

setup(name='pygp',
      version='0.0.1',
      author='Matthew W. Hoffman',
      author_email='mwh30@cam.ac.uk',
      description='A Python package for inference with Gaussian processes',
      url='http://github.com/mwhoffman/pygp',
      license='Simplified BSD',
      packages=find_packages(),
      package_data={'': ['*.txt', '*.npz']},
      install_requires=['numpy', 'scipy', 'matplotlib', 'mwhutils'])
