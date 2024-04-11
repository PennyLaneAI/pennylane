# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the Qubitization Operator template
"""

import pytest
import pennylane as qml
from pennylane import numpy as np

from pennylane.templates.subroutines.qubitization import _positive_coeffs_hamiltonian

@pytest.mark.parametrize(
    "hamiltonian, expected_unitaries",
    (
        #TODO: Waiting to fix this bug: https://github.com/PennyLaneAI/pennylane/issues/5498
        #(qml.dot(np.array([1, -1, 2]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]), [qml.PauliX(0), qml.PauliY(0)@qml.GlobalPhase(np.pi), qml.PauliZ(0)]),
        (qml.dot(np.array([1., 1., 2.]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]),
        #(qml.dot(np.array([-1, -1, 2]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]), [qml.PauliX(0)@qml.GlobalPhase(np.pi), qml.PauliY(0)@qml.GlobalPhase(np.pi), qml.PauliZ(0)]),
),
)
def test_positive_coeffs_hamiltonian(hamiltonian, expected_unitaries):
    """Tests that the function _positive_coeffs_hamiltonian correctly transforms the Hamiltonian"""

    new_coeffs, new_unitaries = _positive_coeffs_hamiltonian(hamiltonian)

    assert np.allclose(new_coeffs, np.abs(hamiltonian.terms()[0]))

    for i in range(len(new_unitaries)):
        assert qml.equal(expected_unitaries[i], new_unitaries[i])


def test_template_definition():
    """Tests that the Qubitization template is correctly defined.
    Based on eq.(65): https://arxiv.org/pdf/2204.11890.pdf"""

    H = qml.dot([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])
    lambda_ = sum([abs(term) for term in H.terms()[0]])

    diag_H = np.diag(qml.matrix(H))
    diag_Q = np.diag(qml.matrix(qml.Qubitization(H, control=[3, 4]), wire_order=[3, 4, 0, 1, 2])[:8, :8])
    assert np.allclose(diag_H, diag_Q * lambda_)

@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_legacy_new_opmath():

    coeffs, ops = [0.1, -0.3, -0.3], [qml.X(0), qml.Z(1), qml.Y(0) @ qml.Z(2)]

    H1 = qml.dot(coeffs, ops)
    matrix_H1 = qml.matrix(qml.Qubitization(H1, control=[3, 4]), wire_order=[3, 4, 0, 1, 2])

    H2 = qml.Hamiltonian(coeffs, ops)
    matrix_H2 = qml.matrix(qml.Qubitization(H2, control=[3, 4]), wire_order=[3, 4, 0, 1, 2])

    assert np.allclose(matrix_H1, matrix_H2)


@pytest.mark.parametrize(
    "hamiltonian, expected_decomposition",
    (
        #TODO: Waiting to fix this bug: https://github.com/PennyLaneAI/pennylane/issues/5498
        #(qml.dot(np.array([1, -1, 2]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]), [qml.PauliX(0), qml.PauliY(0)@qml.GlobalPhase(np.pi), qml.PauliZ(0)]),
        (qml.dot(np.array([1., 1.]), [qml.PauliX(0), qml.PauliZ(0)]), [qml.AmplitudeEmbedding(np.array([1., 1.])/np.sqrt(2), wires=[1]), qml.Select(ops=(qml.X(0), qml.Z(0),), control=[1]), qml.adjoint(qml.AmplitudeEmbedding(np.array([1., 1.])/np.sqrt(2), wires=[1])), qml.FlipSign((0,), wires=[1]), qml.GlobalPhase(np.pi, wires=[1])]),
        #(qml.dot(np.array([-1, -1, 2]), [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]), [qml.PauliX(0)@qml.GlobalPhase(np.pi), qml.PauliY(0)@qml.GlobalPhase(np.pi), qml.PauliZ(0)]),
),
)
def test_decomposition(hamiltonian, expected_decomposition):
    """Tests that the Qubitization template is correctly decomposed."""

    decomposition = qml.Qubitization.compute_decomposition(hamiltonian=hamiltonian, control=[1])

    for i in range(len(decomposition)):
        assert qml.equal(decomposition[i], expected_decomposition[i])

# def test_lightning_qubit(): #TODO: qml.AmplitudeEmbedding in the middle of the circuit is not supported in lightning

# def test_gradient_integration(): #TODO: Issue extracting coeffs from Hamiltonian










