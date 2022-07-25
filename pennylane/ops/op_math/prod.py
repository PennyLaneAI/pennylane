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
This file contains the implementation of the Prod class which contains logic for
computing the product between operations.
"""
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix


def prod(op1, op2):
    """Represent the tensor product (or matrix product) between operators."""
    return Prod(op1, op2)


class Prod(Operator):
    """Arithmetic operator subclass representing the scalar product of an
    operator with the given scalar."""

    def __init__(self, *operators, do_queue=True, id=None):

        self.operators = operators

        combined_wires = qml.wires.Wires.all_wires([op.wires for op in operators])
        combined_params = [op.parameters for op in operators]
        super().__init__(
            *combined_params, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Prod"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([f"{f}" for f in self.operators])

    @property
    def num_wires(self):
        return len(self.wires)

    def terms(self):  # is this method necessary for this class?
        return [1.0], [self]

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

        def matrix_gen(operators, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in operators:
                yield expand_matrix(op.matrix(), op.wires, wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return self._prod(matrix_gen(self.operators, wire_order=wire_order))

    @staticmethod
    def _prod(mats_gen, dtype=None, cast_like=None):
        """Multiply matrices together"""
        res = None
        for i, mat in enumerate(mats_gen):  # In efficient method (should group by wires like in tensor class)
            res = mat if i == 0 else math.dot(res, mat)

        if dtype is not None:              # additional casting logic
            res = math.cast(res, dtype)
        if cast_like is not None:
            res = math.cast_like(res, cast_like)

        return res
