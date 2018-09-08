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
"""OpenQML parameterised variable class"""
import copy
import collections

import numpy as np


def _get_nested(seq, t):
    """Multidimensional sequence indexing"""
    if len(t) > 1:
        return get_nested(seq[t[0]], t[1:])
    return seq[t[0]]

class Variable:
    """Stores the parameter reference.

    Represents a free device parameter (with a non-fixed value).
    Each time the device is executed, it is given a vector of
    parameter values. Variable is essentially an index into that vector.

    Args:
      idx (int): parameter index >= 0
      val (int or complex or float): initial value of the variable (optional)
    """
    def __init__(self, idx, name=None, val=None):
        self.idx = idx  #: int: parameter index
        self.name = name
        self.val = val
        self.mult = 1.0  #: float: parameter scalar multiplier
        self.dim = np.asarray(self.val).shape

    def __str__(self):
        temp = ' * {}'.format(self.mult) if self.mult != 1.0 else ''
        return 'Variable {}: name = {} value = {}{}'.format(self.idx, self.name, self.val, temp)

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

    @staticmethod
    def map(par, values):
        """Mapping function for gate parameters.
        Replaces Variables with their actual values.

        Args:
            par (Sequence[float, int, ParRef]): parameter values to map, each either
                a fixed immediate value or a reference to a free parameter
            values (Sequence[float, int]): values for the free parameters

        Returns:
            list[float, int]: mapped parameters
        """
        return [values[p.idx] * p.mult if isinstance(p, Variable) else p for p in par]

    def __getitem__(self, idx):
        """nested sequence indexing"""
        if isinstance(self.val, collections.Sequence):
            return get_nested(self.val, tuple(idx))

        if isinstance(self.val, np.ndarray):
            return self.val[idx]

        raise IndexError("Variable type does not support indexing.")
