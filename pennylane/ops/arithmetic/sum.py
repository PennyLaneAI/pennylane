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
This file contains the implementation of the Sum class which contains logic for
computing the sum of operations.
"""
import warnings
import numpy as np
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, expand_matrix


def sum(*summands):
    """Compute the sum of the provided terms"""
    return Sum(*summands)  # a wire order is required when combining operators of varying sizes


class Sum(Operator):
    """Arithmetic operator subclass representing the sum of operators"""

    def __init__(self, *summands, do_queue=True, id=None):

        if len(summands) < 2:
            raise ValueError(f"Require at least two operators to sum; got {len(summands)}")

        self.summands = summands

        combined_wires = qml.wires.Wires.all_wires([s.wires for s in summands])
        combined_params = [s.parameters for s in summands]
        super().__init__(
            *combined_params, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Sum"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " + ".join([f"{f}" for f in self.summands])

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def is_hermitian(self):
        """If all of the terms in the sum are hermitian, then the Sum is hermitian."""
        return all([s.is_hermitian for s in self.summands])

    def terms(self):
        return [1.0]*len(self.summands), self.summands

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""

        def matrix_gen(summands, wire_order=None):
            """Helper function to construct a generator of matrices"""
            for op in summands:
                yield expand_matrix(op.matrix(), op.wires, wire_order=wire_order)

        if wire_order is None:
            wire_order = self.wires

        return self._sum(matrix_gen(self.summands, wire_order))

    def eigvals(self, wire_order=None):
        """Get eigvals of the the Sum of operators"""
        if not self.is_hermitian:
            raise qml.operation.EigvalsUndefinedError  # eigvals for non-hermitian obs can be imaginar
        else:
            return super().eigvals()

    def _sum(self, mats_gen, dtype=None, cast_like=None):
        """Super inefficient Sum method just as a proof of concept"""
        res = None
        for i, mat in enumerate(mats_gen):
            res = mat if i == 0 else math.add(res, mat)

        if dtype is not None:                     # additional casting logic
            res = math.cast(res, dtype)
        if cast_like is not None:
            res = math.cast_like(res, cast_like)

        return res
