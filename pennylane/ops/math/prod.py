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
from functools import reduce


class Prod(qml.operation.Operator):
    """Arithmetic operator subclass representing the product of two operators."""

    def __init__(self, *factors, do_queue=True, id=None):

        self.hyperparameters["factors"] = factors
        combined_wires = qml.wires.Wires.all_wires([f.wires for f in factors])
        self.hyperparameters["combined_wires"] = combined_wires

        super().__init__(
            *[f.parameters for f in factors], wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = "Prod(" + ", ".join([f"{f}" for f in factors]) + ")"

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([f"{f}" for f in self.hyperparameters['factors']])

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, factors=None, **hyperparameters):
        return [f for f in factors]

    @staticmethod
    def compute_matrix(*params, factors=None, combined_wires=None, **hyperparams):
        m = factors[0].get_matrix(wire_order=combined_wires)
        for f in factors[1:]:
            m @ f.get_matrix(wire_order=combined_wires)
        return m


def prod(*factors):
    return Prod(*factors)
