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
Unit tests for functions needed for operator and observable arithmetic.
"""
import pennylane as qml
import pytest
from pennylane import numpy as np
from pennylane.hf.transform import _pauli_mult, _return_pauli, simplify


@pytest.mark.parametrize(
    ("p1", "p2", "p_ref"),
    [
        (
            [(0, "X"), (1, "Y")],  # X_0 @ Y_1
            [(0, "X"), (2, "Y")],  # X_0 @ Y_2
            ([(2, "Y"), (1, "Y")], 1.0),  # X_0 @ Y_1 @ X_0 @ Y_2
        ),
    ],
)
def test_pauli_mult(p1, p2, p_ref):
    r"""Test that _pauli_mult returns the correct operator."""
    result = _pauli_mult(p1, p2)

    assert result == p_ref


@pytest.mark.parametrize(
    ("symbol", "operator"),
    [
        ("X", qml.PauliX),
        ("Y", qml.PauliY),
        ("Z", qml.PauliZ),
    ],
)
def test_return_pauli(symbol, operator):
    r"""Test that_return_pauli returns the correct operator."""
    p = _return_pauli(symbol)
    assert p is operator


@pytest.mark.parametrize(
    ("hamiltonian", "result"),
    [
        (
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)]
            ),
            qml.Hamiltonian(np.array([1.0]), [qml.PauliX(0) @ qml.PauliY(1)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.5, -0.5]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)],
            ),
            qml.Hamiltonian([], []),
        ),
        (
            qml.Hamiltonian(
                np.array([0.0, -0.5]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
            ),
            qml.Hamiltonian(np.array([-0.5]), [qml.PauliX(0) @ qml.PauliZ(1)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.25, 0.25, 0.25, -0.25]),
                [
                    qml.PauliX(0) @ qml.PauliY(1),
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.PauliX(0) @ qml.PauliY(1),
                    qml.PauliX(0) @ qml.PauliY(1),
                ],
            ),
            qml.Hamiltonian(
                np.array([0.25, 0.25]),
                [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)],
            ),
        ),
    ],
)
def test_simplify(hamiltonian, result):
    r"""Test that simplify returns the correct hamiltonian."""
    h = simplify(hamiltonian)
    assert h.compare(result)
