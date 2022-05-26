# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This file contains the implementation of the SProd class which contains logic for
computing the scalar product of operations.
"""
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix


def s_prod(scalar, operator):
    """Associate an operator with a scalar to represent scalar multiplication."""
    return SProd(scalar, operator)


class SProd(Operator):
    """Arithmetic operator subclass representing the scalar product of an
    operator with the given scalar."""

    def __init__(self, scalar, operator, do_queue=True, id=None):
        # Add validation checks for size / shape / type of scalar and operator

        if isinstance(operator, self.__class__):  # SProd(0.5, SProd(3, PauliX)) = SProd(1.5, PauliX)
            self.op = operator.op
            self.scalar = scalar * operator.scalar
        else:
            self.op = operator, self.scalar = scalar

        super().__init__(
            operator.parameters, wires=operator.wires, do_queue=do_queue, id=id
        )
        self._name = "SProd"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.scalar}*({self.op})"

    @property
    def num_wires(self):
        return len(self.wires)

    def terms(self):  # is this method necessary for this class?
        return [self.scalar], [self.op]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        if wire_order is None:
            wire_order = self.wires
        return self._sprod(self.op.matrix(wire_order=wire_order), self.scalar)

    @staticmethod
    def _sprod(mat, scalar, dtype=None, cast_like=None):
        """Multiply matrix with scalar"""
        res = math.mult(scalar, mat)

        if dtype is not None:              # additional casting logic
            res = math.cast(res, dtype)
        if cast_like is not None:
            res = math.cast_like(res, cast_like)

        return res
