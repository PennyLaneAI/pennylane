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
from pennylane.pauli import pauli_sentence, PauliWord, PauliSentence


test_hamiltonians = [
    np.array([[2.5, -0.5], [-0.5, 2.5]]),
    np.array(np.diag([0, 0, 0, 1])),
    np.array([[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]),
]

test_general_matrix = [
    np.array([[2.5, -0.5, 1.0], [-0.5, 2.5, -1j]]),
    np.array(np.diag([0, 0, 0, 0, 1])),
    np.array(
        [
            [-2, -2 + 1j, -2, -2, -1],
            [-2 - 1j, 0, 0, -1, -1j],
            [-2, 0, -2, -1, 2],
            [-2, -1, -1, 0, 2j],
        ]
    ),
]

test_diff_matrix = [[[-2, -2 + 1j]], [[-2, -2 + 1j], [-1, -1j]]]


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
        _, obs_list = qml.pauli_decompose(H, hide_identity=True).terms()
        tensors = filter(lambda obs: isinstance(obs, Tensor), obs_list)

        for tensor in tensors:
            all_identities = all(isinstance(o, Identity) for o in tensor.obs)
            no_identities = not any(isinstance(o, Identity) for o in tensor.obs)
            assert all_identities or no_identities

    def test_hide_identity_true_all_identities(self):
        """Tests that the all identity operator remains even with hide_identity = True."""
        H = np.eye(4)
        _, obs_list = qml.pauli_decompose(H, hide_identity=True).terms()
        tensors = filter(lambda obs: isinstance(obs, Tensor), obs_list)

        for tensor in tensors:
            assert all(isinstance(o, Identity) for o in tensor.obs)

    @pytest.mark.parametrize("hide_identity", [True, False])
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_observable_types(self, hamiltonian, hide_identity):
        """Tests that the Hamiltonian decomposes into a linear combination of tensors,
        the identity matrix, and Pauli matrices."""
        allowed_obs = (Tensor, Identity, PauliX, PauliY, PauliZ)

        _, decomposed_obs = qml.pauli_decompose(hamiltonian, hide_identity).terms()
        assert all((isinstance(o, allowed_obs) for o in decomposed_obs))

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_result_length(self, hamiltonian):
        """Tests that tensors are composed of a number of terms equal to the number
        of qubits."""
        _, decomposed_obs = qml.pauli_decompose(hamiltonian).terms()
        n = int(np.log2(len(hamiltonian)))

        tensors = filter(lambda obs: isinstance(obs, Tensor), decomposed_obs)
        assert all(len(tensor.obs) == n for tensor in tensors)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_decomposition(self, hamiltonian):
        """Tests that pauli_decompose successfully decomposes Hamiltonians into a
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
        """Test that wire ordering is working"""
        wire_order1 = ["a", 0]
        wire_order2 = ["auxiliary", "working"]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        for wire_order in (wire_order1, wire_order2):
            h = qml.pauli_decompose(hamiltonian, wire_order=wire_order)
            ps = qml.pauli_decompose(hamiltonian, pauli=True, wire_order=wire_order)

            assert set(ps.wires) == set(wire_order)
            assert h.wires.toset() == set(wire_order)

    def test_wire_error(self):
        """Test that error for incorrect number of wires is raised"""
        wire_order = [0]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        with pytest.raises(
            ValueError, match="number of wires 1 is not compatible with the number of qubits 2"
        ):
            qml.pauli_decompose(hamiltonian, wire_order=wire_order)

        with pytest.raises(
            ValueError, match="number of wires 1 is not compatible with the number of qubits 2"
        ):
            qml.pauli_decompose(hamiltonian, pauli=True, wire_order=wire_order)


class TestPhasedDecomposition:
    """Tests the pauli_decompose_with_phase function"""

    @pytest.mark.parametrize("hamiltonian", [np.ones((4, 2)), np.ones((2, 4))])
    def test_wrong_shape_non_square(self, hamiltonian):
        """Tests that an exception is raised if the Hamiltonian does not have
        the correct shape"""
        with pytest.raises(
            ValueError,
            match="The matrix should be square",
        ):
            qml.pauli_decompose_with_phase(hamiltonian)

    @pytest.mark.parametrize("hamiltonian", [np.ones((5, 5)), np.ones((3, 3))])
    def test_wrong_shape_non_power_two(self, hamiltonian):
        """Tests that an exception is raised if the Hamiltonian does not have
        the correct shape"""
        with pytest.raises(
            ValueError,
            match="Dimension of the matrix should be a power of 2",
        ):
            qml.pauli_decompose_with_phase(hamiltonian)

    def test_hide_identity_true(self):
        """Tests that there are no Identity observables in the tensor products
        when hide_identity=True"""
        H = np.array(np.diag([0, 0, 0, 1]))
        _, obs_list = qml.pauli_decompose_with_phase(H, hide_identity=True)
        tensors = filter(lambda obs: isinstance(obs, Tensor), obs_list)

        for tensor in tensors:
            all_identities = all(isinstance(o, Identity) for o in tensor.obs)
            no_identities = not any(isinstance(o, Identity) for o in tensor.obs)
            assert all_identities or no_identities

    def test_hide_identity_true_all_identities(self):
        """Tests that the all identity operator remains even with hide_identity = True."""
        H = np.eye(4)
        _, obs_list = qml.pauli_decompose_with_phase(H, hide_identity=True)
        tensors = filter(lambda obs: isinstance(obs, Tensor), obs_list)

        for tensor in tensors:
            assert all(isinstance(o, Identity) for o in tensor.obs)

    @pytest.mark.parametrize("hide_identity", [True, False])
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_observable_types(self, hamiltonian, hide_identity):
        """Tests that the Hamiltonian decomposes into a linear combination of tensors,
        the identity matrix, and Pauli matrices."""
        allowed_obs = (Tensor, Identity, PauliX, PauliY, PauliZ)

        _, decomposed_obs = qml.pauli_decompose_with_phase(hamiltonian, hide_identity)
        assert all((isinstance(o, allowed_obs) for o in decomposed_obs))

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_result_length(self, hamiltonian):
        """Tests that tensors are composed of a number of terms equal to the number
        of qubits."""
        _, decomposed_obs = qml.pauli_decompose_with_phase(hamiltonian)
        n = int(np.log2(len(hamiltonian)))

        tensors = filter(lambda obs: isinstance(obs, Tensor), decomposed_obs)
        assert all(len(tensor.obs) == n for tensor in tensors)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_decomposition(self, hamiltonian):
        """Tests that pauli_decompose_with_phase successfully decomposes Hamiltonians into a
        linear combination of Pauli matrices"""
        decomposed_coeff, decomposed_obs = qml.pauli_decompose_with_phase(hamiltonian)
        n = range(int(np.log2(len(hamiltonian))))

        linear_comb = sum(
            [
                decomposed_coeff[i] * qml.matrix(o, wire_order=n)
                for i, o in enumerate(decomposed_obs)
            ]
        )
        assert np.allclose(hamiltonian, linear_comb)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_to_paulisentence(self, hamiltonian):
        """Test that a PauliSentence is returned if the kwarg paulis is set to True"""
        ps = qml.pauli_decompose_with_phase(hamiltonian, pauli=True)
        num_qubits = int(np.log2(len(hamiltonian)))

        assert isinstance(ps, qml.pauli.PauliSentence)
        assert np.allclose(hamiltonian, ps.to_mat(range(num_qubits)))

    @pytest.mark.parametrize("hide_identity", [True, False])
    @pytest.mark.parametrize("matrix", test_general_matrix)
    def test_observable_types_general(self, matrix, hide_identity):
        """Tests that the matrix decomposes into a linear combination of tensors,
        the identity matrix, and Pauli matrices."""
        shape = matrix.shape
        num_qubits = int(np.ceil(np.log2(max(shape))))
        allowed_obs = (Tensor, Identity, PauliX, PauliY, PauliZ)

        decomposed_coeff, decomposed_obs = qml.pauli_decompose_with_phase(
            matrix, hide_identity, padding=True
        )

        assert all((isinstance(o, allowed_obs) for o in decomposed_obs))

        linear_comb = sum(
            [
                decomposed_coeff[i] * qml.matrix(o, wire_order=range(num_qubits))
                for i, o in enumerate(decomposed_obs)
            ]
        )
        assert np.allclose(matrix, linear_comb[: shape[0], : shape[1]])

        if not hide_identity:
            tensors = filter(lambda obs: isinstance(obs, Tensor), decomposed_obs)
            assert all(len(tensor.obs) == num_qubits for tensor in tensors)

    @pytest.mark.parametrize("matrix", test_general_matrix)
    def test_to_paulisentence_general(self, matrix):
        """Test that a PauliSentence is returned if the kwarg paulis is set to True"""
        shape = matrix.shape
        ps = qml.pauli_decompose_with_phase(matrix, padding=True, pauli=True)
        num_qubits = int(np.ceil(np.log2(max(shape))))

        assert isinstance(ps, qml.pauli.PauliSentence)
        assert np.allclose(matrix, ps.to_mat(range(num_qubits))[: shape[0], : shape[1]])

    def test_wire_order(self):
        """Test wire order is working as intended"""
        wire_order1 = ["a", 0]
        wire_order2 = ["auxiliary", "working"]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        for wire_order in (wire_order1, wire_order2):
            c, o = qml.pauli_decompose_with_phase(hamiltonian, wire_order=wire_order)
            h = qml.Hamiltonian(qml.math.real(c), o)

            ps = qml.pauli_decompose_with_phase(hamiltonian, wire_order=wire_order, pauli=True)

            assert set(ps.wires) == set(wire_order)
            assert h.wires.toset() == set(wire_order)

    def test_wire_error(self):
        """Test incorrect wire order throws error"""

        wire_order = [0]
        hamiltonian = np.array(
            [[-2, -2 + 1j, -2, -2], [-2 - 1j, 0, 0, -1], [-2, 0, -2, -1], [-2, -1, -1, 0]]
        )

        with pytest.raises(
            ValueError, match="number of wires 1 is not compatible with the number of qubits 2"
        ):
            qml.pauli_decompose_with_phase(hamiltonian, wire_order=wire_order)

    @pytest.mark.parametrize("pauli", [True, False])
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_active_queing_context(self, hamiltonian, pauli):
        """Test that decompositions are not queued"""

        with qml.queuing.AnnotatedQueue() as q1:
            qml.pauli_decompose_with_phase(hamiltonian, pauli=pauli, padding=True)

        with qml.queuing.AnnotatedQueue() as q2:
            qml.pauli_decompose(hamiltonian, pauli=pauli)

        assert not q1.queue and not q2.queue

    # Multiple interfaces will be tested with math module
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("matrix", test_diff_matrix)
    def test_differentiability(self, matrix):
        """Test differentiability for pauli_decompose_with_phase"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(A):
            coeffs, _ = qml.pauli_decompose_with_phase(A, padding=True)
            qml.RX(qml.math.real(coeffs[2]), 0)
            return qml.expval(qml.PauliZ(0))

        # NumPy Interface
        grad_numpy = qml.grad(circuit)(qml.numpy.array(matrix))

        # Jax Interface
        import jax

        grad_jax = jax.grad(circuit, argnums=(0))(jax.numpy.array(matrix))

        # PyTorch Interface
        import torch

        A = torch.tensor(matrix, requires_grad=True)
        result = circuit(A)
        result.backward()
        grad_torch = A.grad

        # Tensorflow Interface
        import tensorflow as tf

        A = tf.Variable(qml.numpy.array(matrix))
        with tf.GradientTape() as tape:
            loss = circuit(A)
        grad_tflow = tape.gradient(loss, A)

        # Comparisons - note: https://github.com/google/jax/issues/9110
        assert qml.math.allclose(grad_numpy, grad_jax)
        assert qml.math.allclose(grad_torch, grad_tflow)
        assert qml.math.allclose(grad_numpy, qml.math.conjugate(grad_torch))


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
            qml.Hamiltonian([], []),
            PauliSentence(),
        ),
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
            qml.sum(qml.s_prod(2, qml.PauliZ(wires=0)), qml.PauliX(wires=1)),
            PauliSentence({PauliWord({0: "Z"}): 2, PauliWord({1: "X"}): 1}),
        ),
        (
            qml.sum(
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
            qml.sum(
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

    @pytest.mark.parametrize("op, ps", operator_ps)
    def test_operator_private_ps(self, op, ps):
        """Test that a correct pauli sentence is computed when passing an arithmetic operator and not
        relying on the saved op._pauli_rep attribute."""
        assert qml.pauli.conversion._pauli_sentence(op) == ps  # pylint: disable=protected-access

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
