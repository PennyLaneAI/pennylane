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


class Pow(qml.operation.Operator):
    """Arithmetic operator subclass representing the power of an operator."""

    def __init__(self, op, exponent, do_queue=True, id=None):

        self.base_op = op
        self.exponent = exponent

        super().__init__(*op.parameters, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"Pow[{op}, {exponent}]"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.base_op}**{self.exponent}"

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def base_ops(self):
        """List: constituent operations of this arithmetic operation"""
        return [self.base_op]

    def decomposition(self):

        try:
            # extract custom decomposition
            # for now, we wrap this in a list to resemble the output of other decomposition methods
            # in future, decomposition should return an Operation everywhere, and PennyLane should know
            # how to "unpack" a Prod operator
            return [self.base_op.pow(self.exponent)]

        except qml.operation.OperatorPropertyUndefined:

            # default behaviour is defined for some cases
            if self.exponent == 0:
                return []
            if isinstance(self.exponent, int):
                if self.exponent > 0:
                    return [self.base_op] * self.exponent

            raise qml.operation.DecompositionUndefinedError

    def get_matrix(self, wire_order=None):
        if wire_order is None:
            wire_order = self.wires
        return self.base_op.get_matrix(wire_order=wire_order) ** self.exponent  # check if this works for non-integer exponents
