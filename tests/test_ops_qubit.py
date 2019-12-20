# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the available built-in discrete-variable quantum operations.
"""
from functools import reduce

import pytest
import numpy as np

from pennylane.ops import qubit
from pennylane.plugins.default_qubit import Rot3


class TestHadamard:
    """Test functions for Hadamard class."""
    def test_hadamard_matrix(self, tol):
        """Test the Hadamard matrix representation"""
        hadamard = qubit.Hadamard(0)
        matrix = hadamard.matrix()
        true_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        assert np.allclose(matrix, true_matrix, atol=tol, rtol=0)

    def test_hadamard_diagonalization(self, tol):
        """Test the Hadamard diagonalizing_gates function; returned operations
        should transform the Hadamard gate into the Z-gate.

        Only for testing diagonalization based on Rot().
        """
        hadamard = qubit.Hadamard(0)
        zgate = np.array([[1, 0], [0, -1]], dtype=complex)

        gate_list = []
        for op in hadamard.diagonalizing_gates():
            params = op.params
            gate_list.append(Rot3(*params))

        operation = reduce(np.dot, gate_list)

        diag_mat = np.conj(operation).T @ hadamard.matrix() @ operation

        assert np.allclose(zgate, diag_mat, atol=tol, rtol=0)
