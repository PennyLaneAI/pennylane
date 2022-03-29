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


class Prod(qml.operation.Operator):
    """Arithmetic operator subclass representing the product of operators."""

    def __init__(self, *factors, do_queue=True, id=None):

        self.factors = factors
        combined_wires = qml.wires.Wires.all_wires([f.wires for f in factors])
        combined_params = [f.parameters for f in factors]
        super().__init__(
            *combined_params, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Prod[" + ", ".join([f"{f}" for f in factors]) + "]"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([f"{f}" for f in self.factors])

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def base_ops(self):
        """List: constituent operations of this arithmetic operation"""
        return self.factors

    def decomposition(self):
        return [f for f in self.factors]

    def get_matrix(self, wire_order=None):
        if wire_order is None:
            wire_order = self.wires
        m = self.factors[0].get_matrix(wire_order=wire_order)
        for f in self.factors[1:]:
            m @ f.get_matrix(wire_order=wire_order)
        return m

