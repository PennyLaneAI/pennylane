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

import pennylane
from pennylane import numpy as np

from pennylane.ops import qubit


def test_pauli_y_matrix():
    r""" Test that PauliY class has the correct matrix representation.
    """
    true_matrix = np.array([[0, -1j], [1j, 0]])
    assert np.allclose(qubit.PauliY.matrix(), true_matrix)


def test_pauli_y_diagonalization():
    r""" Test that PauliY class returns the correct eigensystem.
    """
    # Eigenvectors are ordered from smallest to largest; eigenvectors are
    # ordered similarly as columns of a matrix.
    eigenvalue_matrix = np.diag([-1, 1])
    eigenvector_matrix = (1 / np.sqrt(2)) * np.array([[-1, -1], [1j, -1j]])
    assert np.allclose(qubit.PauliY.diagonalizing_gates()[0], eigenvalue_matrix)
    assert np.allclose(qubit.PauliY.diagonalizing_gates()[1], eigenvector_matrix)
