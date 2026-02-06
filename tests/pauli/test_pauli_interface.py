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

import pennylane as qp
from pennylane.pauli import pauli_word_prefactor

ops_factors = (
    (qp.PauliX(0), 1),
    (qp.PauliY(1), 1),
    (qp.PauliZ("a"), 1),
    (qp.Identity(0), 1),
    (qp.PauliX(0) @ qp.PauliY(1), 1),
    (qp.PauliX(0) @ qp.PauliY(0), 1j),
    (qp.Hamiltonian([-1.23], [qp.PauliZ(0)]), -1.23),
    (qp.prod(qp.PauliX(0), qp.PauliY(1)), 1),
    (qp.prod(qp.X(0), qp.Y(0)), 1j),
    (qp.s_prod(1.23, qp.s_prod(-1j, qp.PauliZ(0))), -1.23j),
)


@pytest.mark.parametrize("op, true_prefactor", ops_factors)
def test_pauli_word_prefactor(op, true_prefactor):
    """Test that we can accurately determine the prefactor"""
    assert pauli_word_prefactor(op) == true_prefactor


ops = (
    qp.Hadamard(0),
    qp.Hadamard(0) @ qp.PauliZ(1),
    qp.Hamiltonian([], []),
    qp.Hamiltonian([1.23, 0.45], [qp.PauliX(0) @ qp.PauliY(1), qp.PauliZ(1)]),
    qp.prod(qp.PauliX(0), qp.Hadamard(1)),
    qp.prod(qp.sum(qp.PauliX(0), qp.PauliY(0)), qp.PauliZ(1)),
    qp.s_prod(1.23, qp.sum(qp.PauliX(0), qp.PauliY(0))),
)


@pytest.mark.parametrize("op", ops)
def test_pauli_word_prefactor_raises_error(op):
    """Test that an error is raised when the operator provided is not a valid PauliWord."""
    with pytest.raises(ValueError, match="Expected a valid Pauli word, got"):
        pauli_word_prefactor(op)
