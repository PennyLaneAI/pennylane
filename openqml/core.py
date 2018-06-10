# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Core classes
============

**Module name:** :mod:`openqml.core`

.. currentmodule:: openqml.core


Classes
-------

.. autosummary::
   Optimizer
"""

#from numpy import pi, cos, sin, exp, sqrt, arctan, arccosh, sign, arctan2, arcsinh, cosh, tanh, ndarray, all, arange
from .circuit import ParRef


class Optimizer:
    """Quantum circuit optimizer.

    Args:
      pname (str): name of plugin to load
    """
    def __init__(self):
        pass

    def __str__(self):
        """String representation."""
        return self.__class__.__name__
