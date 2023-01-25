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
from functools import partial
import numpy as np
from scipy.linalg import expm
import pytest
import pennylane as qml
from pennylane.ops.qubit.special_unitary import (
    pauli_basis,
    pauli_words,
    special_unitary_matrix,
    _pauli_letters,
    _pauli_matrices,
    _params_to_pauli_basis,
)
from pennylane.wires import Wires


mapped_params_cases = [
    (1, [0.2], "Z", [0.0, 0.0, 0.2]),
    (1, [0.2], ["Y"], [0.0, 0.2, 0.0]),
    (1, [0.2, 0.1], ["X", "Z"], [0.2, 0.0, 0.1]),
    (1, [0.2, 0.1], ["Z", "X"], [0.1, 0.0, 0.2]),
    (2, [0.9, 0.3, 0.2], ["ZZ", "XX", "YY"], [0] * 4 + [0.3] + [0] * 4 + [0.2] + [0] * 4 + [0.9]),
    (1, [[0.2, 0.1], [0.4, 0.9]], ["Z", "X"], [[0.1, 0.0, 0.2], [0.9, 0.0, 0.4]]),
    (
        2,
        [[0.9, 0.3, 0.2], [0.4, 0.1, 0.7]],
        ["ZZ", "XX", "YY"],
        [
            [0] * 4 + [0.3] + [0] * 4 + [0.2] + [0] * 4 + [0.9],
            [0] * 4 + [0.1] + [0] * 4 + [0.7] + [0] * 4 + [0.4],
        ],
    ),
]

faulty_params_cases = [
    ([0.2], ["X", "Z"]),
    ([0.6, 0.2], ["X"]),
    ([0.6, 0.2], ["X", "X"]),
    ([0.2], []),
    ([], []),
]


class TestPauliUtils:
    """Test the utility functions ``pauli_basis`` and ``pauli_words``."""

    def test_pauli_basis_letters(self):
        """Test that the hardcoded Pauli letters and matrices match the PennyLane
        convention regarding order and prefactors."""
        assert _pauli_letters == "IXYZ"
        for op, mat in zip(
            [qml.Identity(0), qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)], _pauli_matrices
        ):
            assert np.allclose(op.matrix(), mat)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pauli_basis(self, n):
        """Test that the Pauli basis matrices are correct."""
        basis = pauli_basis(n)
        d = 4**n - 1
        assert basis.shape == (d, 2**n, 2**n)
        assert np.allclose(basis, basis.conj().transpose([0, 2, 1]))
        assert all(np.allclose(np.eye(2**n), b @ b) for b in basis)

    def test_pauli_basis_raises_too_few_wires(self):
        """Test that pauli_basis raises an error if less than one wire is given."""
        with pytest.raises(ValueError, match="Require at least one"):
            _ = pauli_basis(0)

    def test_pauli_basis_raises_too_many_wires(self):
        """Test that pauli_basis raises an error if too many wires are given."""
        with pytest.raises(ValueError, match="Creating the Pauli basis tensor"):
            _ = pauli_basis(8)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pauli_words(self, n):
        """Test that the Pauli words are correct."""
        words = pauli_words(n)
        d = 4**n - 1
        assert len(words) == d  # There are d words
        assert len(set(words)) == d  # The words are unique
        assert all(len(w) == n for w in words)  # The words all have length n

        # The words consist of I, X, Y, Z, all appear if n>1
        expected_letters = {"I", "X", "Y", "Z"} if n > 1 else {"X", "Y", "Z"}
        assert set("".join(words)) == expected_letters

        # The words are sorted lexicographically
        assert sorted(words) == words

    @pytest.mark.parametrize("x, words", faulty_params_cases)
    def test_params_to_pauli_basis_errors(self, x, words):
        """Test that an error is raised if there are no Pauli words given,
        if the number of words does not match the number of parameters, or if
        the Pauli words are not unique."""
        with pytest.raises(ValueError, match="Expected as many unique Pauli words as parameters"):
            _params_to_pauli_basis(x, words)

    def test_params_to_pauli_basis_error_identity(self):
        """Test that an error is raised if the identity is among the Pauli words."""
        with pytest.raises(ValueError, match="The identity is not a valid"):
            _params_to_pauli_basis([0.1], ["II"])
        with pytest.raises(ValueError, match="The identity is not a valid"):
            _params_to_pauli_basis([0.1, 0.6], ["Z", "I"])

    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_params_to_pauli_basis(self, n, x, words, exp):
        """Test that parameters are mapped correctly to a Pauli basis coordinate vector
        with _params_to_pauli_basis."""
        # pylint:disable=unused-argument
        theta = _params_to_pauli_basis(x, words)
        assert qml.math.shape(theta) == qml.math.shape(exp)
        assert qml.math.allclose(theta, exp)

    @pytest.mark.jax
    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_params_to_pauli_basis_jax(self, n, x, words, exp):
        """Test that parameters in JAX are mapped correctly to a Pauli basis coordinate vector
        with _params_to_pauli_basis and that it is differentiable."""
        import jax

        x = jax.numpy.array(x)
        theta = _params_to_pauli_basis(x, words)
        assert qml.math.shape(theta) == qml.math.shape(exp)
        assert qml.math.allclose(theta, exp)
        jac = jax.jacobian(_params_to_pauli_basis)(x, words)
        assert qml.math.shape(jac) == qml.math.shape(exp) + qml.math.shape(x)
        sorting = np.argsort(words)
        if qml.math.ndim(x) == 1:
            ids = np.where(exp)[0]
            for i, idx in enumerate(ids):
                assert qml.math.allclose(jac[:, sorting[i]], np.eye(4**n - 1)[idx])
        else:
            ids = np.where(exp[0])[0]
            for i, idx in enumerate(ids):
                for batch_idx in range(qml.math.shape(x)[0]):
                    assert qml.math.allclose(
                        jac[batch_idx, :, batch_idx, sorting[i]], np.eye(4**n - 1)[idx]
                    )

    @pytest.mark.torch
    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_params_to_pauli_basis_torch(self, n, x, words, exp):
        """Test that parameters in Torch are mapped correctly to a Pauli basis coordinate vector
        with _params_to_pauli_basis and that it is differentiable."""
        import torch

        x = torch.tensor(x, requires_grad=True)
        theta = _params_to_pauli_basis(x, words)
        assert qml.math.shape(theta) == qml.math.shape(exp)
        assert qml.math.allclose(theta, exp)
        jac = torch.autograd.functional.jacobian(partial(_params_to_pauli_basis, words=words), x)
        assert qml.math.shape(jac) == qml.math.shape(exp) + qml.math.shape(x)
        sorting = np.argsort(words)
        if qml.math.ndim(x) == 1:
            ids = np.where(exp)[0]
            for i, idx in enumerate(ids):
                assert qml.math.allclose(jac[:, sorting[i]], np.eye(4**n - 1)[idx])
        else:
            ids = np.where(exp[0])[0]
            for i, idx in enumerate(ids):
                for batch_idx in range(qml.math.shape(x)[0]):
                    assert qml.math.allclose(
                        jac[batch_idx, :, batch_idx, sorting[i]], np.eye(4**n - 1)[idx]
                    )

    @pytest.mark.autograd
    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_params_to_pauli_basis_autograd(self, n, x, words, exp):
        """Test that parameters in Autograd are mapped correctly to a Pauli basis coordinate vector
        with _params_to_pauli_basis and that it is differentiable."""
        x = qml.numpy.array(x, requires_grad=True)
        if qml.math.ndim(x) > 1:
            pytest.skip("Autograd scatter_element_add does not support broadcasting.")
        theta = _params_to_pauli_basis(x, words)
        assert qml.math.shape(theta) == qml.math.shape(exp)
        assert qml.math.allclose(theta, exp)
        jac = qml.jacobian(_params_to_pauli_basis, argnum=0)(x, words)
        assert qml.math.shape(jac) == qml.math.shape(exp) + qml.math.shape(x)
        sorting = np.argsort(words)
        ids = np.where(exp)[0]
        for i, idx in enumerate(ids):
            assert qml.math.allclose(jac[:, sorting[i]], np.eye(4**n - 1)[idx])

    @pytest.mark.tf
    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_params_to_pauli_basis_tf(self, n, x, words, exp):
        """Test that parameters in Tensorflow are mapped correctly to a Pauli basis coordinate vector
        with _params_to_pauli_basis and that it is differentiable."""
        import tensorflow as tf

        x = tf.Variable(x)
        theta = _params_to_pauli_basis(x, words)
        assert qml.math.shape(theta) == qml.math.shape(exp)
        assert qml.math.allclose(theta, exp)
        with tf.GradientTape() as t:
            out = _params_to_pauli_basis(x, words)
        jac = t.jacobian(out, x)
        assert qml.math.shape(jac) == qml.math.shape(exp) + qml.math.shape(x)
        sorting = np.argsort(words)
        if qml.math.ndim(x) == 1:
            ids = np.where(exp)[0]
            for i, idx in enumerate(ids):
                assert qml.math.allclose(jac[:, sorting[i]], np.eye(4**n - 1)[idx])
        else:
            ids = np.where(exp[0])[0]
            for i, idx in enumerate(ids):
                for batch_idx in range(qml.math.shape(x)[0]):
                    assert qml.math.allclose(
                        jac[batch_idx, :, batch_idx, sorting[i]], np.eye(4**n - 1)[idx]
                    )


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


class TestSpecialUnitaryMatrix:
    """Test the special_unitary_matrix utility function."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("seed", [214, 2491, 8623])
    def test_special_unitary_matrix_random(self, n, seed):
        """Test that ``special_unitary_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters."""
        np.random.seed(seed)
        d = 4**n - 1
        theta = np.random.random(d)
        matrix = special_unitary_matrix(theta, n)
        assert matrix.shape == (2**n, 2**n)
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2**n))

    @pytest.mark.parametrize("seed", [214, 8623])
    def test_special_unitary_matrix_random_many_wires(self, seed):
        """Test that ``special_unitary_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters and more than 5 wires."""
        np.random.seed(seed)
        n = 6
        d = 4**n - 1
        theta = np.random.random(d)
        matrix = special_unitary_matrix(theta, n)
        assert matrix.shape == (2**n, 2**n)
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2**n))

    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("seed", [214, 2491, 8623])
    def test_special_unitary_matrix_random_broadcasted(self, n, seed):
        """Test that ``special_unitary_matrix`` returns a correctly-shaped
        unitary matrix for broadcasted random input parameters."""
        np.random.seed(seed)
        d = 4**n - 1
        theta = np.random.random((2, d))
        matrix = special_unitary_matrix(theta, n)
        assert matrix.shape == (2, 2**n, 2**n)
        assert all(np.allclose(m @ m.conj().T, np.eye(2**n)) for m in matrix)
        separate_matrices = [special_unitary_matrix(t, n) for t in theta]
        assert qml.math.allclose(separate_matrices, matrix)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_special_unitary_matrix_single_param(self, n):
        """Test that ``special_unitary_matrix`` returns a Pauli rotation matrix for
        inputs with a single non-zero parameter, and that the parameter mapping
        matches the lexicographical ordering of ``pauli_words``."""
        d = 4**n - 1
        words = pauli_words(n)
        for word, theta in zip(words, np.eye(d)):
            x = 0.2142
            matrix = special_unitary_matrix(x * theta, n)
            paulirot_matrix = qml.PauliRot(-2 * x, word, wires=list(range(n))).matrix()
            assert np.allclose(matrix @ matrix.conj().T, np.eye(2**n))
            assert np.allclose(paulirot_matrix, matrix)

    @pytest.mark.parametrize("n, theta, expected", special_matrix_cases)
    def test_special_unitary_matrix_special_cases(self, n, theta, expected):
        """Tests that ``special_unitary_matrix`` returns the correct matrices
        for a few specific cases."""
        matrix = special_unitary_matrix(theta, n)
        assert qml.math.allclose(matrix, expected)

    def test_special_unitary_matrix_special_cases_broadcasted(self):
        """Tests that ``special_unitary_matrix`` returns the correct matrices
        for a few specific cases broadcasted together."""
        _, thetas, exp = zip(*special_matrix_cases[1:])
        theta = np.stack(thetas)
        matrix = special_unitary_matrix(theta, 2)
        assert qml.math.allclose(matrix, np.stack(exp))


theta_1 = np.array([0.4, 0.1, 0.1])
theta_2 = np.array([0.4, 0.1, 0.1, 0.6, 0.2, 0.3, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.1, 0.5, 0.2])
n_and_theta = [(1, theta_1), (2, theta_2)]


class TestSpecialUnitary:
    """Tests for the Operation ``SpecialUnitary``."""

    @pytest.mark.parametrize("n, x, words, exp", mapped_params_cases)
    def test_param_mapping_alias(self, n, x, words, exp):
        """Test that using the parameter mapping of sparse input
        parameters to initialize SpecialUnitary works."""
        wires = list(range(n))
        op_map = qml.SpecialUnitary(x, wires, words=words)
        op_full_par = qml.SpecialUnitary(exp, wires)

        assert qml.equal(op_map, op_full_par)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition(self, n, theta):
        """Test that SpecialUnitary falls back to QubitUnitary."""

        wires = list(range(n))
        decomp = qml.SpecialUnitary.compute_decomposition(theta, wires, n)
        decomp2 = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1 == len(decomp2)
        assert decomp[0].name == "QubitUnitary" == decomp2[0].name
        assert decomp[0].wires == Wires(wires) == decomp2[0].wires
        mat = special_unitary_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)
        assert np.allclose(decomp2[0].data[0], mat)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_broadcasted(self, n, theta):
        """Test that the broadcasted SpecialUnitary falls back to QubitUnitary."""
        theta = np.outer([0.2, 1.0, -0.3], theta)
        wires = list(range(n))

        decomp = qml.SpecialUnitary.compute_decomposition(theta, wires, n)
        decomp2 = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1 == len(decomp2)
        assert decomp[0].name == "QubitUnitary" == decomp2[0].name
        assert decomp[0].wires == Wires(wires) == decomp2[0].wires

        mat = special_unitary_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)
        assert np.allclose(decomp2[0].data[0], mat)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_matrix_representation(self, n, theta, tol):
        """Test that the matrix representation is defined correctly"""
        wires = list(range(n))
        res_static = qml.SpecialUnitary.compute_matrix(theta, n)
        res_dynamic = qml.SpecialUnitary(theta, wires).matrix()
        expected = special_unitary_matrix(theta, n)
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("n, theta, expected", special_matrix_cases)
    def test_matrix_representation_special_cases(self, n, theta, expected):
        """Tests that ``special_unitary_matrix`` returns the correct matrices
        for a few specific cases."""
        wires = list(range(n))
        res_static = qml.SpecialUnitary.compute_matrix(theta, n)
        res_dynamic = qml.SpecialUnitary(theta, wires).matrix()
        assert qml.math.allclose(res_static, expected)
        assert qml.math.allclose(res_dynamic, expected)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_matrix_representation_broadcasted(self, n, theta, tol):
        """Test that the matrix representation is defined correctly for
        a broadcasted SpecialUnitary."""
        theta = np.outer([0.2, 1.0, -0.3], theta)
        wires = list(range(n))
        res_static = qml.SpecialUnitary.compute_matrix(theta, n)
        res_dynamic = qml.SpecialUnitary(theta, wires).matrix()
        expected = special_unitary_matrix(theta, n)
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_matrix_unitarity(self, n):
        """Test that the matrix of SpecialUnitary is unitary."""
        wires = list(range(n))
        d = 4**n - 1
        theta = np.random.random(d)
        U = qml.SpecialUnitary(theta, wires).matrix()
        assert qml.math.allclose(U.conj().T @ U, np.eye(2**n))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_matrix_PauliRot(self, n):
        """Test that the matrix of SpecialUnitary matches the matrix
        of a PauliRot operation if only one term is active in the parameters."""
        wires = list(range(n))
        d = 4**n - 1
        words = pauli_words(n)
        prefactors = np.random.random(d)
        thetas = prefactors * np.eye(d)
        for theta, pref, word in zip(thetas, prefactors, words):
            U = qml.SpecialUnitary(theta, wires)
            rot = qml.PauliRot(-2 * pref, word, wires)
            assert qml.math.allclose(U.matrix(), rot.matrix())

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_matrix_broadcasting(self, theta, n, batch_size):
        """Test that the matrix of SpecialUnitary works with broadcasting and is unitary."""
        wires = list(range(n))
        theta = np.outer(np.arange(batch_size), theta)
        U = qml.SpecialUnitary(theta, wires).matrix()
        assert all(qml.math.allclose(_U, special_unitary_matrix(_t, n)) for _U, _t in zip(U, theta))

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
            state = special_unitary_matrix(x, 1) @ jnp.array([1, 0])
            return jnp.abs(state) ** 2

        jac = jax.jacobian(circuit)(theta)
        expected_jac = jax.jacobian(comparison)(theta)
        assert np.allclose(jac, expected_jac)

    # The JAX version of scipy.linalg.expm does not support broadcasting.
    @pytest.mark.xfail
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
            state = special_unitary_matrix(x, 1) @ jnp.array([1, 0])
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
                special_unitary_matrix(x, 1),
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
