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
This module contains the functions needed for estimating the number of logical qubits and
non-Clifford gates for quantum algorithms in first quantization using a plane-wave basis.
"""
from pennylane import numpy as np
from pennylane.operation import Operation, AnyWires


class FQ(Operation):
    """Contains the functionality for estimating the number of non-Clifford gates and logical qubits
    for quantum algorithms in first quantization using a plane-wave basis.

    Args:
        n (int): number of basis states
        eta (int): number of electrons
        omega (float): unit cell volume
        error (float): target error in the algorithm
        charge (int): total electric charge of the system
        br (int): number of bits for ancilla qubit rotation
    """

    def __init__(
        self,
        n,
        eta,
        omega,
        error=0.0016,
        charge=0,
        br=7,
    ):
        self.n = n
        self.eta = eta
        self.omega = omega
        self.error = error
        self.charge = charge
        self.br = br

        self.lamb = self.norm(self.eta, self.n, self.omega, self.error, self.br, self.charge)

        self.gates = self.gate_cost(
            self.n, self.eta, self.omega, self.error, self.lamb, self.br, self.charge
        )
        self.qubits = self.qubit_cost(
            self.n, self.eta, self.omega, self.error, self.lamb, self.charge
        )

    num_wires = AnyWires
    grad_method = None
