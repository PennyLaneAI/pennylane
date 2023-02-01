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
Unit tests for the SpecialUnitary operation and its utility functions.
"""
# pylint:disable=import-outside-toplevel
import numpy as np
from scipy.linalg import expm
import pytest
import pennylane as qml
from pennylane.ops.qubit.special_unitary import (
    pauli_basis_matrices,
    pauli_basis_strings,
    _pauli_letters,
    _pauli_matrices,
)
from pennylane.wires import Wires


class TestPauliUtils:
    """Test the utility functions ``pauli_basis_matrices`` and ``pauli_basis_strings``."""

    def test_pauli_letters(self):
        """Test that the hardcoded Pauli letters and matrices match the PennyLane
        convention regarding order and prefactors."""
        assert _pauli_letters == "IXYZ"
        for op, mat in zip(
            [qml.Identity(0), qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)], _pauli_matrices
        ):
            assert np.allclose(op.matrix(), mat)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pauli_basis_matrices(self, n):
        """Test that the Pauli basis matrices are correct."""
        basis = pauli_basis_matrices(n)
        d = 4**n - 1
        assert basis.shape == (d, 2**n, 2**n)
        assert np.allclose(basis, basis.conj().transpose([0, 2, 1]))
        assert all(np.allclose(np.eye(2**n), b @ b) for b in basis)

    def test_pauli_basis_matrices_raises_too_few_wires(self):
        """Test that pauli_basis_matrices raises an error if less than one wire is given."""
        with pytest.raises(ValueError, match="Require at least one"):
            _ = pauli_basis_matrices(0)

    def test_pauli_basis_matrices_raises_too_many_wires(self):
        """Test that pauli_basis_matrices raises an error if too many wires are given."""
        with pytest.raises(ValueError, match="Creating the Pauli basis tensor"):
            _ = pauli_basis_matrices(8)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pauli_basis_strings(self, n):
        """Test that the Pauli words are correct."""
        words = pauli_basis_strings(n)
        d = 4**n - 1
        assert len(words) == d  # There are d words
        assert len(set(words)) == d  # The words are unique
        assert all(len(w) == n for w in words)  # The words all have length n

        # The words consist of I, X, Y, Z, all appear if n>1
        expected_letters = {"I", "X", "Y", "Z"} if n > 1 else {"X", "Y", "Z"}
        assert set("".join(words)) == expected_letters

        # The words are sorted lexicographically
        assert sorted(words) == words


eye = np.eye(15)
special_matrix_cases = [
    (1, [0.2, 0.1, 2.1], expm([[2.1j, 0.2j + 0.1], [0.2j - 0.1, -2.1j]])),
    (
        2,
        [0.2, 0.3, 0.1] + [0] * 12,
        np.kron(np.eye(2), expm([[0.1j, 0.2j + 0.3], [0.2j - 0.3, -0.1j]])),
    ),
    (
        2,
        eye[0] + eye[3] + eye[4],
        np.kron(qml.RX(-2, 0).matrix(), qml.RX(-2, 0).matrix()) @ qml.IsingXX(-2, [0, 1]).matrix(),
    ),
    (2, 0.6 * (eye[4] + eye[9]), qml.IsingXY(2.4, [0, 1]).matrix()),
    (2, 0.8 * eye[-1], qml.IsingZZ(-1.6, [0, 1]).matrix()),
]

theta_1 = np.array([0.4, 0.1, 0.1])
theta_2 = np.array([0.4, 0.1, 0.1, 0.6, 0.2, 0.3, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.1, 0.5, 0.2])
n_and_theta = [(1, theta_1), (2, theta_2)]

interfaces = [
    None,
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tf", marks=pytest.mark.tf),
]


class TestSpecialUnitary:
    """Tests for the Operation ``SpecialUnitary``."""

    @staticmethod
    def interface_array(x, interface):
        """Create a trainable array of x in the specified interface."""
        if interface is None:
            return x
        if interface == "autograd":
            return qml.numpy.array(x)
        if interface == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)
            return jax.numpy.array(x)
        if interface == "torch":
            import torch

            return torch.tensor(x)
        if interface == "tf":
            import tensorflow as tf

            return tf.Variable(x)
        return None

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("seed", [214, 2491, 8623])
    def test_compute_matrix_random(self, n, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters."""
        np.random.seed(seed)
        d = 4**n - 1
        theta = np.random.random(d)
        theta = self.interface_array(theta, interface)
        matrices = [
            qml.SpecialUnitary(theta, list(range(n))).matrix(),
            qml.SpecialUnitary.compute_matrix(theta, n),
        ]
        I = self.interface_array(np.eye(2**n), interface)
        for matrix in matrices:
            if interface == "torch":
                matrix = matrix.detach().numpy()
            assert matrix.shape == (2**n, 2**n)
            assert np.allclose(matrix @ qml.math.conj(qml.math.T(matrix)), I)

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("seed", [214, 8623])
    def test_compute_matrix_random_many_wires(self, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters and more than 5 wires."""
        np.random.seed(seed)
        n = 6
        d = 4**n - 1
        theta = np.random.random(d)
        theta = self.interface_array(theta, interface)
        matrices = [
            qml.SpecialUnitary(theta, list(range(n))).matrix(),
            qml.SpecialUnitary.compute_matrix(theta, n),
        ]
        I = self.interface_array(np.eye(2**n, dtype=np.complex128), interface)
        for matrix in matrices:
            if interface == "torch":
                matrix = matrix.detach().numpy()
            assert matrix.shape == (2**n, 2**n)
            assert np.allclose(matrix @ qml.math.conj(qml.math.T(matrix)), I)

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("seed", [214, 2491, 8623])
    def test_compute_matrix_random_broadcasted(self, n, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for broadcasted random input parameters."""
        np.random.seed(seed)
        d = 4**n - 1
        theta = np.random.random((2, d))
        separate_matrices = [qml.SpecialUnitary.compute_matrix(t, n) for t in theta]
        theta = self.interface_array(theta, interface)
        matrices = [
            qml.SpecialUnitary(theta, list(range(n))).matrix(),
            qml.SpecialUnitary.compute_matrix(theta, n),
        ]
        I = self.interface_array(np.eye(2**n), interface)
        for matrix in matrices:
            if interface == "torch":
                matrix = matrix.detach().numpy()
            assert matrix.shape == (2, 2**n, 2**n)
            assert all(np.allclose(m @ qml.math.conj(qml.math.T(m)), I) for m in matrix)
            assert qml.math.allclose(separate_matrices, matrix)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_compute_matrix_single_param(self, n):
        """Test that ``compute_matrix`` returns a Pauli rotation matrix for
        inputs with a single non-zero parameter, and that the parameter mapping
        matches the lexicographical ordering of ``pauli_basis_strings``."""
        d = 4**n - 1
        words = pauli_basis_strings(n)
        for word, theta in zip(words, np.eye(d)):
            x = 0.2142
            matrices = [
                qml.SpecialUnitary(x * theta, list(range(n))).matrix(),
                qml.SpecialUnitary.compute_matrix(x * theta, n),
            ]
            paulirot_matrix = qml.PauliRot(-2 * x, word, list(range(n))).matrix()
            for matrix in matrices:
                assert np.allclose(matrix @ matrix.conj().T, np.eye(2**n))
                assert np.allclose(paulirot_matrix, matrix)

    @pytest.mark.parametrize("n, theta, expected", special_matrix_cases)
    def test_compute_matrix_special_cases(self, n, theta, expected):
        """Tests that ``compute_matrix`` returns the correct matrices
        for a few specific cases."""
        matrices = [
            qml.SpecialUnitary(theta, list(range(n))).matrix(),
            qml.SpecialUnitary.compute_matrix(theta, n),
        ]
        for matrix in matrices:
            assert qml.math.allclose(matrix, expected)

    def test_compute_matrix_special_cases_broadcasted(self):
        """Tests that ``compute_matrix`` returns the correct matrices
        for a few specific cases broadcasted together."""
        _, thetas, exp = zip(*special_matrix_cases[1:])
        n = 2
        theta = np.stack(thetas)
        matrices = [
            qml.SpecialUnitary(theta, list(range(n))).matrix(),
            qml.SpecialUnitary.compute_matrix(theta, n),
        ]
        for matrix in matrices:
            assert qml.math.allclose(matrix, np.stack(exp))

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition(self, n, theta):
        """Test that SpecialUnitary falls back to QubitUnitary."""

        wires = list(range(n))
        decomp = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1
        assert decomp[0].name == "QubitUnitary"
        assert decomp[0].wires == Wires(wires)
        mat = qml.SpecialUnitary.compute_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_broadcasted(self, n, theta):
        """Test that the broadcasted SpecialUnitary falls back to QubitUnitary."""
        theta = np.outer([0.2, 1.0, -0.3], theta)
        wires = list(range(n))

        decomp = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1
        assert decomp[0].name == "QubitUnitary"
        assert decomp[0].wires == Wires(wires)

        mat = qml.SpecialUnitary.compute_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_adjoint(self, theta, n):
        """Test the adjoint of SpecialUnitary."""
        wires = list(range(n))
        U = qml.SpecialUnitary(theta, wires)
        U_dagger = qml.adjoint(qml.SpecialUnitary)(theta, wires)
        U_dagger_inplace = qml.SpecialUnitary(theta, wires).adjoint()
        U_minustheta = qml.SpecialUnitary(-theta, wires)
        assert qml.math.allclose(U.matrix(), U_dagger.matrix().conj().T)
        assert qml.math.allclose(U.matrix(), U_dagger_inplace.matrix().conj().T)
        assert qml.math.allclose(U_minustheta.matrix(), U_dagger.matrix())

    @pytest.mark.parametrize(
        "theta, n", [(np.ones(4), 1), (9.421, 2), (np.ones((5, 2, 1)), 1), (np.ones((5, 16)), 2)]
    )
    def test_wrong_input_shape(self, theta, n):
        """Test that an error is raised if the parameters of SpecialUnitary have the wrong shape."""
        wires = list(range(n))
        with pytest.raises(ValueError, match="Expected the parameters to have"):
            _ = qml.SpecialUnitary(theta, wires)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that the SpecialUnitary operation works
        within a QNode that uses the JAX JIT"""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=1, shots=None)

        theta = jnp.array(theta_1)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.SpecialUnitary(x, 0)
            return qml.probs(wires=[0])

        def comparison(x):
            state = qml.SpecialUnitary.compute_matrix(x, 1) @ jnp.array([1, 0])
            return jnp.abs(state) ** 2

        jac = jax.jacobian(circuit)(theta)
        expected_jac = jax.jacobian(comparison)(theta)
        assert np.allclose(jac, expected_jac)

    @pytest.mark.jax
    def test_jax_jit_broadcasted(self):
        """Test that the SpecialUnitary operation works
        within a QNode that uses the JAX JIT and broadcasting."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=1, shots=None)

        theta = jnp.outer(jnp.array([-0.4, 0.1, 1.0]), theta_1)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.SpecialUnitary(x, 0)
            return qml.probs(wires=[0])

        def comparison(x):
            state = qml.SpecialUnitary.compute_matrix(x, 1) @ jnp.array([1, 0])
            return jnp.abs(state) ** 2

        jac = jax.jacobian(circuit)(theta)
        expected_jac = jax.jacobian(comparison)(theta)
        assert np.allclose(jac, expected_jac)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf_function(self):
        """Test that the SpecialUnitary operation works
        within a QNode that uses TensorFlow autograph"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1, shots=None)

        @tf.function
        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.SpecialUnitary(x, 0)
            return qml.expval(qml.PauliX(0))

        theta = tf.Variable(theta_1)

        with tf.GradientTape() as tape:
            loss = circuit(theta)

        jac = tape.jacobian(loss, theta)

        def comparison(x):
            state = qml.math.tensordot(
                qml.SpecialUnitary.compute_matrix(x, 1),
                tf.constant([1, 0], dtype=tf.complex128),
                axes=[[1], [0]],
            )
            return qml.math.tensordot(
                qml.math.conj(state),
                qml.math.tensordot(qml.PauliX(0).matrix(), state, axes=[[1], [0]]),
                axes=[[0], [0]],
            )

        with tf.GradientTape() as tape:
            loss = comparison(theta)

        expected = tape.jacobian(loss, theta)
        assert np.allclose(jac, expected)
