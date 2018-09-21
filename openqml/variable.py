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

import copy
import collections

import numpy as np


def _get_nested(seq, t):
    """Multidimensional sequence indexing"""
    if len(t) > 1:
        return get_nested(seq[t[0]], t[1:])
    return seq[t[0]]

class Variable:
    """Free parameter reference.

    Represents a free quantum circuit parameter (with a non-fixed value),
    times an optional scalar multiplier.
    Each time the circuit is executed, it is given a vector of
    parameter values. Variable is essentially an index into that vector.

    Args:
      idx  (int): parameter index >= 0
      name (str): name of the variable (optional)
    """
    free_param_values = None  #: array[float]: current free parameter values, set in :meth:`QNode.evaluate`

    def __init__(self, idx, name=None):
        self.idx = idx    #: int: parameter index
        self.name = name  #: str: parameter name  FIXME unused?
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
        return self.free_param_values[self.idx] * self.mult
