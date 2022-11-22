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
from pennylane.operation import Tensor
from pennylane.ops import Identity, PauliX, PauliY, PauliZ


test_hamiltonians = [
    np.array([[2.5, -0.5], [-0.5, 2.5]]),
    np.array(np.diag([0, 0, 0, 1])),
    np.array([[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]),
]


class TestDecomposition:
    """Tests the pauli_decompose function"""

    @pytest.mark.parametrize("hamiltonian", [np.ones((3, 3)), np.ones((4, 2)), np.ones((2, 4))])
    def test_wrong_shape(self, hamiltonian):
        """Tests that an exception is raised if the Hamiltonian does not have
        the correct shape"""
        with pytest.raises(
            ValueError,
            match="The matrix should have shape",
        ):
            qml.pauli_decompose(hamiltonian)

    def test_not_hermitian(self):
        """Tests that an exception is raised if the Hamiltonian is not Hermitian, i.e.
        equal to its own conjugate transpose"""
        with pytest.raises(ValueError, match="The matrix is not Hermitian"):
            qml.pauli_decompose(np.array([[1, 2], [3, 4]]))

    def test_hide_identity_true(self):
        """Tests that there are no Identity observables in the tensor products
        when hide_identity=True"""
        H = np.array(np.diag([0, 0, 0, 1]))
        coeff, obs_list = qml.pauli_decompose(H, hide_identity=True).terms()
        tensors = filter(lambda obs: isinstance(obs, Tensor), obs_list)

        for tensor in tensors:
            all_identities = all(isinstance(o, Identity) for o in tensor.obs)
            no_identities = not any(isinstance(o, Identity) for o in tensor.obs)
            assert all_identities or no_identities

    @pytest.mark.parametrize("hide_identity", [True, False])
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_observable_types(self, hamiltonian, hide_identity):
        """Tests that the Hamiltonian decomposes into a linear combination of tensors,
        the identity matrix, and Pauli matrices."""
        allowed_obs = (Tensor, Identity, PauliX, PauliY, PauliZ)

        decomposed_coeff, decomposed_obs = qml.pauli_decompose(hamiltonian, hide_identity).terms()
        assert all([isinstance(o, allowed_obs) for o in decomposed_obs])

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_result_length(self, hamiltonian):
        """Tests that tensors are composed of a number of terms equal to the number
        of qubits."""
        decomposed_coeff, decomposed_obs = qml.pauli_decompose(hamiltonian).terms()
        n = int(np.log2(len(hamiltonian)))

        tensors = filter(lambda obs: isinstance(obs, Tensor), decomposed_obs)
        assert all(len(tensor.obs) == n for tensor in tensors)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_decomposition(self, hamiltonian):
        """Tests that decompose_hamiltonian successfully decomposes Hamiltonians into a
        linear combination of Pauli matrices"""
        decomposed_coeff, decomposed_obs = qml.pauli_decompose(hamiltonian).terms()

        linear_comb = sum([decomposed_coeff[i] * o.matrix() for i, o in enumerate(decomposed_obs)])
        assert np.allclose(hamiltonian, linear_comb)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_to_paulisentence(self, hamiltonian):
        """Test that a PauliSentence is returned if the kwarg paulis is set to True"""
        ps = qml.pauli_decompose(hamiltonian, pauli=True)
        num_qubits = int(np.log2(len(hamiltonian)))

        assert isinstance(ps, qml.pauli.PauliSentence)
        assert np.allclose(hamiltonian, ps.to_mat(range(num_qubits)))

    def test_wire_order(self):
        wire_order1 = ["a", 0]
        wire_order2 = ["auxiliary", "working"]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        for wire_order in (wire_order1, wire_order2):
            h = qml.pauli_decompose(hamiltonian, wire_order=wire_order)
            ps = qml.pauli_decompose(hamiltonian, pauli=True, wire_order=wire_order)

            assert ps.wires == set(wire_order)
            assert h.wires.toset() == set(wire_order)

    def test_wire_error(self):
        wire_order = [0]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        with pytest.raises(
            ValueError, match="number of wires 1 is not compatible with number of qubits 2"
        ):
            h = qml.pauli_decompose(hamiltonian, wire_order=wire_order)

        with pytest.raises(
            ValueError, match="number of wires 1 is not compatible with number of qubits 2"
        ):
            ps = qml.pauli_decompose(hamiltonian, pauli=True, wire_order=wire_order)
