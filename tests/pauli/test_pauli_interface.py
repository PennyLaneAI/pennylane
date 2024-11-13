# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pauli` interface functions in ``pauli/pauli_interface.py``.
"""
import pytest

import pennylane as qml
from pennylane.pauli import pauli_word_prefactor

ops_factors = (
    (qml.PauliX(0), 1),
    (qml.PauliY(1), 1),
    (qml.PauliZ("a"), 1),
    (qml.Identity(0), 1),
    (qml.PauliX(0) @ qml.PauliY(1), 1),
    (qml.PauliX(0) @ qml.PauliY(0), 1j),
    (qml.Hamiltonian([-1.23], [qml.PauliZ(0)]), -1.23),
    (qml.prod(qml.PauliX(0), qml.PauliY(1)), 1),
    (qml.prod(qml.X(0), qml.Y(0)), 1j),
    (qml.s_prod(1.23, qml.s_prod(-1j, qml.PauliZ(0))), -1.23j),
)


@pytest.mark.parametrize("op, true_prefactor", ops_factors)
def test_pauli_word_prefactor(op, true_prefactor):
    """Test that we can accurately determine the prefactor"""
    assert pauli_word_prefactor(op) == true_prefactor


ops = (
    qml.Hadamard(0),
    qml.Hadamard(0) @ qml.PauliZ(1),
    qml.Hamiltonian([], []),
    qml.Hamiltonian([1.23, 0.45], [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(1)]),
    qml.prod(qml.PauliX(0), qml.Hadamard(1)),
    qml.prod(qml.sum(qml.PauliX(0), qml.PauliY(0)), qml.PauliZ(1)),
    qml.s_prod(1.23, qml.sum(qml.PauliX(0), qml.PauliY(0))),
)


@pytest.mark.parametrize("op", ops)
def test_pauli_word_prefactor_raises_error(op):
    """Test that an error is raised when the operator provided is not a valid PauliWord."""
    with pytest.raises(ValueError, match="Expected a valid Pauli word, got"):
        pauli_word_prefactor(op)
