<<<<<<< HEAD
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
r"""
Contains the SwapTest template.
"""

import pennylane as qml
from pennylane.operation import Operation, AnyWires, Wires
from pennylane.ops import Hadamard, CSWAP
from pennylane.measure import sample


class SwapTest(Operation):
    """A class that implements the SWAPTest"""
    num_wires = AnyWires
    grad_recipe = None

    def __init__(self, ancilla, q_reg1, q_reg2, do_queue=True, id=None):
        wires = [ancilla] + q_reg1 + q_reg2

        if len(q_reg1) != len(q_reg2):
            raise ValueError(f"The two quantum registers must be the same size to compare them via SWAPTest, "
                             f"got: {q_reg1}, {q_reg2}")

        self.ac = ancilla
        self.q_reg1 = q_reg1
        self.q_reg2 = q_reg2

        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 0

    def expand(self):
        with qml.tape.QuantumTape() as tape:
            len_register = len(self.q_reg1)

            Hadamard(wires=self.ac)  # apply hadamard to ancilla qubit
            for i in range(len_register):
                CSWAP(wires=Wires([self.ac, self.q_reg1[i], self.q_reg2[i]]))  # swap qubit registers
            Hadamard(wires=self.ac)  # apply hadamard again

        return tape

    def __call__(self, *args, **kwargs):
        return sample(wires=self.ac)
