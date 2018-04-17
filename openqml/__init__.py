# Copyright 2018 Xanadu Quantum Technologies Inc.
"""
.. _code:

Overview
========

Top-level functions
-------------------

.. autosummary::
   version
"""

__all__ = ['version']


"Version number (major.minor.patch[-label])"
__version__ = '0.0.0'


def version():
    """
    Version number of OpenQML.

    Returns:
      str: package version number
    """
    return __version__
