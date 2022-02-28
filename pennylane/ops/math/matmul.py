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


class MatMul(qml.operation.Operator):
    def __init__(self, left, right, do_queue=True, id=None):

        self.hyperparameters["left"] = left
        self.hyperparameters["right"] = right

        combined_wires = qml.wires.Wires.all_wires([left.wires, right.wires])
        super().__init__(
            *left.parameters, *right.parameters, wires=combined_wires, do_queue=do_queue, id=id
        )
        self._name = f"{right.name} {left.name}"

    def __repr__(self):
        """Constructor-call-like representation."""
        return f"{self.hyperparameters['left']} \n@ {self.hyperparameters['right']}"

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_decomposition(*params, left, right, wires=None, **hyperparameters):
        return [left, right]

    @staticmethod
    def compute_matrix(*params, left, right, **hyperparams):
        # ugly to compute this here again!
        combined_wires = qml.wires.Wires.all_wires([left.wires, right.wires])
        return left.get_matrix(wire_order=combined_wires) @ right.get_matrix(
            wire_order=combined_wires
        )


def matmul(left, right):
    return MatMul(left, right)
