# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.ops.qubit`.
"""
# pylint: disable=protected-access,cell-var-from-loop

import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.ops import qubit


def test_pauli_y_matrix():
    r""" Test that PauliY class has the correct matrix representation.
    """
    expected_matrix = np.array([[0, -1j], [1j, 0]])
    assert np.allclose(qubit.PauliY.matrix(), expected_matrix)


def test_pauli_y_diagonalization():
    r""" Test that the gates diagonalize the PauliY.
    """
    dev = qml.device("default.qubit", wires=1)

    diagonalizing_gates = qubit.PauliY.diagonalizing_gates()
    is_diagonalizing = []

    with qml.utils.OperationRecorder() as rec:
        for gate in diagonalizing_gates:
            matrix_rep = dev._get_operator_matrix(gate.base_name, par=gate.parameters)
            # U^\dag PauliY U should be diagonal; subtract off the diagonal and compare to all 0s
            diagonal = np.conj(matrix_rep.T) @ qubit.PauliY.matrix() @ matrix_rep
            is_diagonalizing.append(
                np.allclose(diagonal - np.diag(np.diag(diagonal)), np.zeros((2, 2)))
            )

    assert np.all(is_diagonalizing)
