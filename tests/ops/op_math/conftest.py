# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines utility classes and functions for testing."""

import numpy as np
from scipy import sparse

import pennylane as qp
from pennylane.core.operator import Operator2

# pylint: disable=unused-argument


class SX2(Operator2):
    """A new SX gate."""

    wire_sizes = (1,)

    is_verified_hermitian = True

    def __init__(self, wires):
        super().__init__(wires)

    @property
    def pauli_rep(self):
        return qp.pauli.PauliSentence(
            {
                qp.pauli.PauliWord({self.wires[0]: "I"}): (0.5 + 0.5j),
                qp.pauli.PauliWord({self.wires[0]: "X"}): (0.5 - 0.5j),
            }
        )

    @staticmethod
    def compute_matrix(wires):
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def compute_sparse_matrix(wires, format="csr"):
        return 0.5 * sparse.csr_matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]).asformat(format=format)

    @staticmethod
    def compute_eigvals(wires):
        return np.array([1, 1j])

    @staticmethod
    def compute_diagonalizing_gates(wires):
        return [qp.Hadamard(wires=wires)]


class RX2(Operator2):
    """A simplified RX operator that inherit from Operator2."""

    dynamic_argnames = ("theta",)

    wire_sizes = (1,)

    is_verified_hermitian = True

    def __init__(self, theta, wires):
        super().__init__(theta, wires)

    @staticmethod
    def compute_matrix(theta, wires):
        return qp.math.array(
            [
                [qp.math.cos(theta / 2), -1j * qp.math.sin(theta / 2)],
                [-1j * qp.math.sin(theta / 2), qp.math.cos(theta / 2)],
            ]
        )

    def generator(self):
        return qp.Hamiltonian([-0.5], [qp.X(wires=self.wires)])

    def adjoint(self):
        return RX2(-self.theta, wires=self.wires)

    def simplify(self):
        theta = self.theta % (4 * np.pi)
        if qp.math.isclose(theta, 0.0):
            return qp.Identity()
        return RX2(theta, wires=self.wires)
