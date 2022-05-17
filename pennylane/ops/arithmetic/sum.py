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
from pennylane.operation import Operator


def sum(*summands):
    """Compute the sum of the provided terms"""
    return Sum(*summands)


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
        self.matrix_cache = None  # reduce overhead by saving matrix representation for a Sum instance

    def __repr__(self):
        """Constructor-call-like representation."""
        return " + ".join([f"{f}" for f in self.summands])

    @property
    def num_wires(self):
        return len(self.wires)

    def terms(self):
        return [1.0]*len(self.summands), self.summands

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis.

        Args:

        Raises:

        Returns:

        """
        if self.matrix_cache is not None:
            return self.matrix_cache

        def matrix_gen(summands):
            """Helper function to construct a generator of matrices"""
            for op in summands:
                try:
                    mat = op.compute_matrix()  # static method to generate matrix rep for op
                except qml.operation.MatrixUndefinedError:
                    mat = op.matrix()  # non-static method to get matrix rep for op
                yield mat

        canonical_matrix = self._sum(matrix_gen(self.summands))
        if wire_order is None or self.wires == qml.Wires(wire_order):
            self.matrix_cache = canonical_matrix
        else:
            self.matrix_cache = qml.operation.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

        return self.matrix_cache

    def _sum(self, mats_gen):
        """Super inefficient Sum method just as a proof of concept"""
        res = np.zeros((2**self.num_wires, 2**self.num_wires), dtype="complex128")  # TODO: fix casting errors here !

        for mat in mats_gen:
            res += mat

        return res
