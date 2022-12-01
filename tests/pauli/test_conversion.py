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
"""Unit tests for utility functions of Pauli arithmetic."""

import pytest

import numpy as np
import pennylane as qml
from pennylane.pauli import pauli_sentence, PauliWord, PauliSentence


test_hamiltonians = [
    np.array([[2.5, -0.5], [-0.5, 2.5]]),
    np.array(np.diag([0, 0, 0, 1])),
    np.array([[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]),
]


class TestPauliSentence:
    """Test the pauli_sentence function."""

    pauli_op_ps = (
        (qml.PauliX(wires=0), PauliSentence({PauliWord({0: "X"}): 1})),
        (qml.PauliY(wires=1), PauliSentence({PauliWord({1: "Y"}): 1})),
        (qml.PauliZ(wires="a"), PauliSentence({PauliWord({"a": "Z"}): 1})),
        (qml.Identity(wires=0), PauliSentence({PauliWord({}): 1})),
    )

    @pytest.mark.parametrize("op, ps", pauli_op_ps)
    def test_pauli_ops(self, op, ps):
        """Test that PL Pauli ops are properly cast to a PauliSentence."""
        assert pauli_sentence(op) == ps

    tensor_ps = (
        (
            qml.PauliX(wires=0) @ qml.PauliZ(wires=1),
            PauliSentence({PauliWord({0: "X", 1: "Z"}): 1}),
        ),
        (qml.PauliX(wires=0) @ qml.Identity(wires=1), PauliSentence({PauliWord({0: "X"}): 1})),
        (
            qml.PauliX(wires=0)
            @ qml.PauliY(wires="a")
            @ qml.PauliZ(wires=1)
            @ qml.Identity(wires="b"),
            PauliSentence({PauliWord({0: "X", "a": "Y", 1: "Z"}): 1}),
        ),
        (qml.PauliX(wires=0) @ qml.PauliY(wires=0), PauliSentence({PauliWord({0: "Z"}): 1j})),
    )

    @pytest.mark.parametrize("op, ps", tensor_ps)
    def test_tensor(self, op, ps):
        """Test that Tensors of Pauli ops are properly cast to a PauliSentence."""
        assert pauli_sentence(op) == ps

    def test_tensor_raises_error(self):
        """Test that Tensors of non-Pauli ops raise error when cast to a PauliSentence."""
        h_mat = np.array([[1, 1], [1, -1]])
        h_op = qml.Hermitian(h_mat, wires=1)
        op = qml.PauliX(wires=0) @ h_op

        with pytest.raises(ValueError, match="Op must be a linear combination of"):
            pauli_sentence(op)

    hamiltonian_ps = (
        (
            qml.Hamiltonian([2], [qml.PauliZ(wires=0)]),
            PauliSentence({PauliWord({0: "Z"}): 2}),
        ),
        (
            qml.Hamiltonian(
                [2, -0.5], [qml.PauliZ(wires=0), qml.PauliX(wires=0) @ qml.PauliZ(wires=1)]
            ),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 2,
                    PauliWord({0: "X", 1: "Z"}): -0.5,
                }
            ),
        ),
        (
            qml.Hamiltonian(
                [2, -0.5, 3.14],
                [
                    qml.PauliZ(wires=0),
                    qml.PauliX(wires=0) @ qml.PauliZ(wires="a"),
                    qml.Identity(wires="b"),
                ],
            ),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 2,
                    PauliWord({0: "X", "a": "Z"}): -0.5,
                    PauliWord({}): 3.14,
                }
            ),
        ),
    )

    @pytest.mark.parametrize("op, ps", hamiltonian_ps)
    def test_hamiltonian(self, op, ps):
        """Test that a Hamiltonian is properly cast to a PauliSentence."""
        assert pauli_sentence(op) == ps

    operator_ps = (
        (
            qml.op_sum(qml.s_prod(2, qml.PauliZ(wires=0)), qml.PauliX(wires=1)),
            PauliSentence({PauliWord({0: "Z"}): 2, PauliWord({1: "X"}): 1}),
        ),
        (
            qml.op_sum(
                qml.s_prod(2, qml.PauliZ(wires=0)),
                -0.5 * qml.prod(qml.PauliX(wires=0), qml.PauliZ(wires=1)),
            ),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 2,
                    PauliWord({0: "X", 1: "Z"}): -0.5,
                }
            ),
        ),
        (
            qml.op_sum(
                qml.s_prod(2, qml.PauliZ(wires=0)),
                -0.5 * qml.prod(qml.PauliX(wires=0), qml.PauliZ(wires="a")),
                qml.s_prod(2.14, qml.Identity(wires=0)),
                qml.Identity(wires="a"),
            ),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 2,
                    PauliWord({0: "X", "a": "Z"}): -0.5,
                    PauliWord({}): 3.14,
                }
            ),
        ),
    )

    @pytest.mark.parametrize("op, ps", operator_ps)
    def test_operator(self, op, ps):
        """Test that PL arithmetic op is properly cast to a PauliSentence."""
        assert pauli_sentence(op) == ps

    error_ps = (
        qml.Hadamard(wires=0),
        qml.Hamiltonian([1, 2], [qml.Projector([0], wires=0), qml.PauliZ(wires=1)]),
        qml.RX(1.23, wires="a") + qml.PauliZ(wires=0),
    )

    @pytest.mark.parametrize("op", error_ps)
    def test_error_not_linear_comb_pauli_words(self, op):
        """Test that a ValueError is raised when trying to cast operators to a PauliSentence
        which are not linear combinations of Pauli words."""
        with pytest.raises(
            ValueError, match="Op must be a linear combination of Pauli operators only, got:"
        ):
            pauli_sentence(op)
