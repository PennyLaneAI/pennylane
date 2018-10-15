# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Quantum circuit parameters
==========================

**Module name:** :mod:`openqml.variable`

.. currentmodule:: openqml.variable


Classes
-------

.. autosummary::
   Variable

----
"""
import logging
import copy
import collections

import numpy as np

logging.getLogger()


class Variable:
    """Free parameter reference.

    Represents a placeholder variable. This can either be a free quantum
    circuit parameter (with a non-fixed value) times an optional scalar multiplier,
    or a placeholder for data/other hard-coded data.

    Each time the circuit is executed, it is given a vector of
    parameter values, and a dictionary of keyword variable values.

    Variable is essentially an index into that vector.

    .. note:: Variables currently do not implement any arithmetic
        operations other than scalar multiplication.

    Args:
      idx  (int): parameter index >= 0
      name (str): name of the variable (optional)
    """
    free_param_values = None  #: array[float]: current free parameter values, set in :meth:`QNode.evaluate`
    kwarg_values = None # dictionary containing the keyword argument values.

    def __init__(self, idx=None, name=None):
        self.idx = idx    #: int: parameter index
        self.name = name  #: str: parameter name
        self.mult = 1     #: int, float: parameter scalar multiplier

    def __str__(self):
        temp = ' * {}'.format(self.mult) if self.mult != 1.0 else ''
        return 'Variable {}: name = {}, {}'.format(self.idx, self.name, temp)

    def __neg__(self):
        """Unary negation."""
        temp = copy.copy(self)
        temp.mult = -temp.mult
        return temp

    def __mul__(self, scalar):
        """Right multiplication by scalars."""
        temp = copy.copy(self)
        temp.mult *= scalar
        return temp

    __rmul__ = __mul__ # """Left multiplication by scalars."""

    @property
    def val(self):
        """Current numerical value of the Variable.

        Returns:
            float: current value of the Variable
        """
        if self.name is None:
            # The variable is a placeholder for a positional argument
            value = self.free_param_values[self.idx] * self.mult
            logging.debug("Positional arg idx: %g val: %f", self.idx, value)
            return value

        # The variable is a placeholder for a keyword argument
        value = self.kwarg_values[self.name][self.idx] * self.mult
        logging.debug("Keyword arg name: %s idx: %g val: %f", self.name, self.idx, value)
        return value
