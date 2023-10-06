# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This submodule contains the template for AQFT.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation


class AQFT(Operation):
    r"""stuff"""
    num_wires = AnyWires
    grad_method = None

    def __init__(self, order, wires=None, id=None):
        wires = qml.wires.Wires(wires)
        self.hyperparameters["order"] = order
        self.hyperparameters["n_wires"] = len(wires)
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(
        wires, n_wires, order
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""stuff"""
        shifts = [2 * np.pi * 2**-i for i in range(2, n_wires + 1)]

        decomp_ops = []
        for i, wire in enumerate(wires):
            decomp_ops.append(qml.Hadamard(wire))
            counter = 0

            for shift, control_wire in zip(shifts[: len(shifts) - i], wires[i + 1 :]):
                op = qml.ControlledPhaseShift(shift, wires=[control_wire, wire])
                decomp_ops.append(op)
                counter = counter + 1
                if counter >= order:
                    break

        first_half_wires = wires[: n_wires // 2]
        last_half_wires = wires[-(n_wires // 2) :]

        for wire1, wire2 in zip(first_half_wires, reversed(last_half_wires)):
            swap = qml.SWAP(wires=[wire1, wire2])
            decomp_ops.append(swap)

        return decomp_ops
