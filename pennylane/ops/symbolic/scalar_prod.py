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
import pennylane as qml


class ScalarProd(qml.operation.Operator):
    """Arithmetic operator subclass representing the product of an operator with a scalar."""

    def __init__(self, scalar, op, do_queue=True, id=None):

        self.scalar = scalar
        self.base_op = op
        super().__init__(*op.parameters, scalar, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"{scalar}  {op.name}"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.scalar} {self.base_op}"

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def base_ops(self):
        """List: constituent operations of this arithmetic operation"""
        return [self.base_op]

    def terms(self):
        return [self.scalar], [self.base_op]

    def get_matrix(self, wire_order=None):
        return self.scalar * self.base_op.get_matrix(wire_order)

    def get_eigvals(self):
        return self.scalar * self.base_op.get_eigvals()
