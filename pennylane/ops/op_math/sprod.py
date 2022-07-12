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
from .symbolicop import SymbolicOp


def s_prod(scalar, operator, do_queue=True, id=None):
    """Associate an operator with a scalar to represent scalar multiplication."""
    return SProd(scalar, operator, do_queue=do_queue, id=id)


def _sprod(mat, scalar, dtype=None, cast_like=None):
    """Multiply matrix with scalar"""
    res = qml.math.multiply(scalar, mat)

    if dtype is not None:  # additional casting logic
        res = qml.math.cast(res, dtype)
    if cast_like is not None:
        res = qml.math.cast_like(res, cast_like)

    return res


class SProd(SymbolicOp):
    """Arithmetic operator subclass representing the scalar product of an
    operator with the given scalar."""

    def __init__(self, scalar, base, do_queue=True, id=None):
        self.scalar = scalar
        super().__init__(base=base, do_queue=do_queue, id=id)
        self._name = "SProd"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.scalar}*({self.base})"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or f"{self.scalar} {self.base.label(decimals=decimals, cache=cache)}"

    @property
    def data(self):
        """The trainable parameters"""
        return [[self.scalar], self.base.data]  # Not sure if this is the best way to deal with this

    @data.setter
    def data(self, new_data):
        self.scalar = new_data[0]
        if len(new_data) > 1:
            self.base.data = new_data[1:]

    @property
    def parameters(self):
        return self.data.copy()

    @property
    def num_params(self):
        return 1 + self.base.num_params

    def terms(self):  # is this method necessary for this class?
        return [self.scalar], [self.base]

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        return self.scalar * self.base.eigvals()

    def sparse_matrix(self, wire_order=None):
        return _sprod(self.base.sparse_matrix(wire_order=wire_order), self.scalar)

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis."""
        return _sprod(self.base.matrix(wire_order=wire_order), self.scalar)

    @property
    def _queue_category(self):  # don't queue scalar prods as they might not be Unitary!
        """Used for sorting objects into their respective lists in `QuantumTape` objects.
        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns: None
        """
        return None
