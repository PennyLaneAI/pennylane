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

# pylint: disable=too-few-public-methods,function-redefined

import pennylane as qml


class Exp(qml.operation.Operator):
    """Arithmetic operator subclass representing the exponential of an operator."""

    def __init__(self, op, do_queue=True, id=None):

        self.hyperparameters["base_op"] = op

        super().__init__(*op.parameters, wires=op.wires, do_queue=do_queue, id=id)
        self._name = f"Exp({op})"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"exp({self.hyperparameters['base_op']})"

    @property
    def num_wires(self):
        return len(self.wires)

    def generator(self):
        return -1j * self.hyperparameters["base_op"]

    @staticmethod
    def compute_matrix(*params, base_op=None):
        return qml.math.expm1(base_op.get_matrix())  # check if this gets dispatched correctly

    @staticmethod
    def compute_eigvals(*params, base_op=None):
        return qml.math.exp(base_op.get_eigvals()) # check if this gets dispatched correctly


def exp(op):
    try:
        # there is a custom version defined
        return op.exp()
    except AttributeError:
        # default to an abstract arithmetic class
        return Exp(op)
