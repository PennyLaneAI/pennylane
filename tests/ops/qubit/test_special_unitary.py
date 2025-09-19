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
import pytest
from scipy.linalg import expm

import pennylane as qml
from pennylane.ops.qubit.special_unitary import (
    TmpPauliRot,
    _pauli_letters,
    _pauli_matrices,
    pauli_basis_matrices,
    pauli_basis_strings,
)
from pennylane.transforms.convert_to_numpy_parameters import _convert_op_to_numpy_data
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


class TestGetOneParameterGenerators:
    """Tests for the effective generators computing function
    get_one_parameter_generators of qml.SpecialUnitary."""

    @staticmethod
    def get_one_parameter_generators(theta, num_wires, interface):
        """Create a SpecialUnitary operation and return its one-parameter group generators."""
        return qml.SpecialUnitary(theta, list(range(num_wires))).get_one_parameter_generators(
            interface
        )

    @pytest.mark.jax
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jax(self, n, use_jit, seed):
        """Test that generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        rng = np.random.default_rng(seed)
        d = 4**n - 1
        theta = jnp.array(rng.random(d))
        fn = (
            jax.jit(self.get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else self.get_one_parameter_generators
        )
        Omegas = fn(theta, n, "jax")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(jnp.allclose(O.conj().T, -O) for O in Omegas)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jax_pauli_generated(self, use_jit):
        """Test that generators match Pauli words."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        n = 1
        fn = (
            jax.jit(self.get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else self.get_one_parameter_generators
        )
        basis = pauli_basis_matrices(n)
        d = 4**n - 1
        for i, (theta, pauli_mat) in enumerate(zip(jnp.eye(d), basis)):
            Omegas = fn(theta, n, "jax")
            assert Omegas.shape == (d, 2**n, 2**n)
            assert all(jnp.allclose(O.conj().T, -O) for O in Omegas)
            assert jnp.allclose(Omegas[i], 1j * pauli_mat)

    @pytest.mark.tf
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tf(self, n):
        """Test that generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        Omegas = self.get_one_parameter_generators(theta, n, "tf")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.tf
    def test_tf_pauli_generated(self):
        """Test that generators match Pauli words."""
        import tensorflow as tf

        n = 1
        basis = pauli_basis_matrices(n)
        d = 4**n - 1
        for i, (theta, pauli_mat) in enumerate(zip(np.eye(d), basis)):
            theta = tf.Variable(theta)
            Omegas = self.get_one_parameter_generators(theta, n, "tf")
            assert Omegas.shape == (d, 2**n, 2**n)
            assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)
            assert qml.math.allclose(Omegas[i], 1j * pauli_mat)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_torch(self, n):
        """Test that generators are computed correctly in Torch."""
        import torch

        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        Omegas = self.get_one_parameter_generators(theta, n, "torch")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.torch
    def test_torch_pauli_generated(self):
        """Test that generators match Pauli words."""
        import torch

        n = 1
        basis = pauli_basis_matrices(n)
        d = 4**n - 1
        for i, (theta, pauli_mat) in enumerate(zip(torch.eye(d, requires_grad=True), basis)):
            Omegas = self.get_one_parameter_generators(theta, n, "torch")
            assert Omegas.shape == (d, 2**n, 2**n)
            assert all(
                qml.math.allclose(qml.math.conj(qml.math.T(O)), -O, atol=3e-8) for O in Omegas
            )
            assert qml.math.allclose(Omegas[i], 1j * pauli_mat)

    def test_raises_autograd(self):
        """Test that computing generators raises an
        error when attempting to use Autograd."""
        op = qml.SpecialUnitary(qml.numpy.ones(3), [0])
        with pytest.raises(NotImplementedError, match="expm is not differentiable in Autograd"):
            op.get_one_parameter_generators("autograd")

    def test_raises_unknown_interface(self):
        """Test that computing generators raises an error when attempting
        to use an unknown interface."""
        with pytest.raises(ValueError, match="The interface test is not supported"):
            self.get_one_parameter_generators(np.ones(3), 1, "test")

    def test_raises_broadcasting(self):
        """Test that computing generators raises an error when attempting
        to use broadcasting."""
        theta = np.random.random((2, 3))
        with pytest.raises(ValueError, match="Broadcasting is not supported"):
            self.get_one_parameter_generators(theta, 1, "dummy")


class TestGetOneParameterGeneratorsDiffability:
    """Tests for the effective generators computing function
    get_one_parameter_generators of qml.SpecialUnitary to be differentiable."""

    @staticmethod
    def get_one_parameter_generators(theta, num_wires, interface):
        """Create a SpecialUnitary operation and return its one-parameter group generators."""
        return qml.SpecialUnitary(theta, list(range(num_wires))).get_one_parameter_generators(
            interface
        )

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    @pytest.mark.parametrize("n", [1, 2])
    def test_jacobian_jax(self, n, use_jit):
        """Test that generators are differentiable in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        d = 4**n - 1
        theta = jnp.array(np.random.random(d), dtype=jnp.complex128)
        fn = (
            jax.jit(self.get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else self.get_one_parameter_generators
        )
        dOmegas = jax.jacobian(fn, holomorphic=True)(theta, n, "jax")
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.tf
    @pytest.mark.parametrize("n", [1, 2])
    def test_jacobian_tf(self, n):
        """Test that generators are differentiable in Tensorflow."""
        import tensorflow as tf

        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        with tf.GradientTape() as t:
            Omegas = self.get_one_parameter_generators(theta, n, "tf")
        dOmegas = t.jacobian(Omegas, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", [1, 2])
    def test_jacobian_torch(self, n):
        """Test that generators are differentiable in Torch."""
        import torch

        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)

        def fn(theta):
            return self.get_one_parameter_generators(theta, n, "torch")

        dOmegas = torch.autograd.functional.jacobian(fn, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    def test_raises_autograd(self):
        """Test that computing generator derivatives raises an error
        when attempting to use Autograd."""
        with pytest.raises(NotImplementedError, match="expm is not differentiable in Autograd"):
            qml.jacobian(self.get_one_parameter_generators)(qml.numpy.ones(3), 1, "autograd")


class TestGetOneParameterCoeffs:
    """Tests for the coefficients of effective generators computing function
    get_one_parameter_coeffs of qml.SpecialUnitary."""

    @pytest.mark.jax
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_jax(self, n):
        """Test that the coefficients of the generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        op = qml.SpecialUnitary(theta, list(range(n)))
        omegas = op.get_one_parameter_coeffs("jax")
        assert omegas.shape == (d, d)
        assert jnp.allclose(omegas.real, 0)

        basis = pauli_basis_matrices(n)
        reconstructed_Omegas = jnp.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = op.get_one_parameter_generators("jax")
        assert jnp.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.tf
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tf(self, n):
        """Test that the coefficients of the generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        op = qml.SpecialUnitary(theta, list(range(n)))
        omegas = op.get_one_parameter_coeffs("tf")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis_matrices(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = op.get_one_parameter_generators("tf")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_torch(self, n):
        """Test that the coefficients of the generators are computed correctly in Torch."""
        import torch

        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        op = qml.SpecialUnitary(theta, list(range(n)))
        omegas = op.get_one_parameter_coeffs("torch")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis_matrices(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = op.get_one_parameter_generators("torch")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    def test_raises_autograd(self):
        """Test that computing coefficient derivatives raises an
        error when attempting to use Autograd."""
        op = qml.SpecialUnitary(qml.numpy.ones(3), [0])
        with pytest.raises(NotImplementedError, match="expm is not differentiable in Autograd"):
            op.get_one_parameter_coeffs("autograd")


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
    def test_compute_matrix_random(self, n, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters."""
        rng = np.random.default_rng(seed)
        d = 4**n - 1
        theta = rng.random(d)
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
    def test_compute_matrix_random_many_wires(self, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for random input parameters and more than 5 wires."""
        rng = np.random.default_rng(seed)
        n = 6
        d = 4**n - 1
        theta = rng.random(d)
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
    def test_compute_matrix_random_broadcasted(self, n, seed, interface):
        """Test that ``compute_matrix`` returns a correctly-shaped
        unitary matrix for broadcasted random input parameters."""
        rng = np.random.default_rng(seed)
        d = 4**n - 1
        theta = rng.random((2, d))
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
        x = 0.2142
        for word, theta in zip(words, np.eye(d)):
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
    def test_decomposition_numpy(self, n, theta):
        """Test that SpecialUnitary falls back to QubitUnitary with NumPy inputs."""
        wires = list(range(n))
        decomp = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1
        assert decomp[0].name == "QubitUnitary"
        assert decomp[0].wires == Wires(wires)
        mat = qml.SpecialUnitary.compute_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_broadcasted_numpy(self, n, theta):
        """Test that the broadcasted SpecialUnitary falls back to QubitUnitary with NumPy inputs."""
        theta = np.outer([0.2, 1.0, -0.3], theta)
        wires = list(range(n))

        decomp = qml.SpecialUnitary(theta, wires).decomposition()

        assert len(decomp) == 1
        assert decomp[0].name == "QubitUnitary"
        assert decomp[0].wires == Wires(wires)

        mat = qml.SpecialUnitary.compute_matrix(theta, n)
        assert np.allclose(decomp[0].data[0], mat)

    @pytest.mark.jax
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_jax(self, n, theta):
        """Test that a trainable SpecialUnitary in JAX
        decomposes into a non-trainable SpecialUnitary and TmpPauliRot ops."""
        import jax

        jax.config.update("jax_enable_x64", True)

        d = 4**n - 1
        words = pauli_basis_strings(n)
        wires = list(range(n))

        def assertion_fn(theta):
            """Wrapper function to allow marking the parameters as trainable in JAX."""
            decomp = qml.SpecialUnitary(theta, wires).decomposition()
            assert len(decomp) == d + 1
            for w, op in zip(words, decomp[:-1]):
                qml.assert_equal(
                    TmpPauliRot(0.0, w, wires=wires),
                    op,
                    check_trainability=False,
                    check_interface=False,
                )
            qml.assert_equal(qml.SpecialUnitary(qml.math.detach(theta), wires=wires), decomp[-1])

            decomp = qml.SpecialUnitary(qml.math.detach(theta), wires).decomposition()
            mat = qml.SpecialUnitary.compute_matrix(qml.math.detach(theta), n)
            assert len(decomp) == 1
            qml.assert_equal(qml.QubitUnitary(mat, wires=wires), decomp[0])

            return theta

        theta = jax.numpy.array(theta)
        jax.jacobian(assertion_fn)(theta)

    @pytest.mark.torch
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_torch(self, n, theta):
        """Test that a trainable SpecialUnitary in Torch
        decomposes into a non-trainable SpecialUnitary and TmpPauliRot ops."""
        import torch

        d = 4**n - 1
        words = pauli_basis_strings(n)
        wires = list(range(n))
        theta = torch.tensor(theta, requires_grad=True)
        decomp = qml.SpecialUnitary(theta, wires).decomposition()
        assert len(decomp) == d + 1
        for w, op in zip(words, decomp[:-1]):
            qml.assert_equal(
                TmpPauliRot(0.0, w, wires=wires),
                op,
                check_trainability=False,
                check_interface=False,
            )
        qml.assert_equal(qml.SpecialUnitary(qml.math.detach(theta), wires=wires), decomp[-1])

        decomp = qml.SpecialUnitary(qml.math.detach(theta), wires).decomposition()
        mat = qml.SpecialUnitary.compute_matrix(qml.math.detach(theta), n)
        assert len(decomp) == 1
        qml.assert_equal(qml.QubitUnitary(mat, wires=wires), decomp[0])

    @pytest.mark.tf
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_tf(self, n, theta):
        """Test that a trainable SpecialUnitary in Tensorflow
        decomposes into a non-trainable SpecialUnitary and TmpPauliRot ops."""
        import tensorflow as tf

        d = 4**n - 1
        words = pauli_basis_strings(n)
        wires = list(range(n))
        theta = tf.Variable(theta)
        with tf.GradientTape():
            decomp = qml.SpecialUnitary(theta, wires).decomposition()
            assert len(decomp) == d + 1
            for w, op in zip(words, decomp[:-1]):
                qml.assert_equal(
                    TmpPauliRot(0.0, w, wires=wires),
                    op,
                    check_trainability=False,
                    check_interface=False,
                )
            qml.assert_equal(qml.SpecialUnitary(qml.math.detach(theta), wires=wires), decomp[-1])

            decomp = qml.SpecialUnitary(qml.math.detach(theta), wires).decomposition()
            mat = qml.SpecialUnitary.compute_matrix(qml.math.detach(theta), n)
            assert len(decomp) == 1
            qml.assert_equal(qml.QubitUnitary(mat, wires=wires), decomp[0])

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

        dev = qml.device("default.qubit", wires=1)

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

        dev = qml.device("default.qubit", wires=1)

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

        dev = qml.device("default.qubit", wires=1)

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

    @pytest.mark.torch
    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ["jax", "torch"])
    def test_large_wire_jacobian_regression(self, interface):
        """Regression test for compute_matrix with num_wires > 5.

        This test specifically addresses a bug where the interface conversion
        was not properly handled for large wire counts, causing tensor type
        mismatches in the itertools.product path vs pauli_basis_matrices path.
        """
        # Use 6 wires to trigger itertools.product path
        num_wires = 6  # This triggers the itertools.product code path (num_wires > 5)

        # Create just 10 parameters for testing - this is sufficient to test interface conversion
        num_params = 10
        theta_np = (
            np.random.randn(num_params) * 0.01 + 0.0j
        )  # crucial for jax to proceed with holomorphic
        theta = qml.math.asarray(theta_np, like=interface, requires_grad=True)

        # This should not raise an error about tensor type mismatches
        matrix = qml.SpecialUnitary.compute_matrix(theta, num_wires=num_wires)

        expected_shape = (2**num_wires, 2**num_wires)
        assert qml.math.shape(matrix) == expected_shape

        # Test that gradients can be computed (this was the main issue https://github.com/PennyLaneAI/pennylane/issues/7583)

        def g(theta):
            return qml.math.real(qml.SpecialUnitary.compute_matrix(theta, num_wires=num_wires))

        jac = qml.math.jacobian(g)(theta)
        assert qml.math.shape(jac) == expected_shape + (num_params,)


@pytest.mark.parametrize("dev_fn", [qml.devices.DefaultQubit])
class TestSpecialUnitaryIntegration:
    """Test that the operation SpecialUnitary is executable and
    differentiable in a QNode context, both with automatic differentiation
    and parameter-shift rules."""

    @staticmethod
    def circuit(x):
        """Test circuit involving a single SpecialUnitary operations."""
        qml.SpecialUnitary(x, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    @staticmethod
    def paulirot_comp_circuit(x, word=None):
        """Test circuit involving a single PauliRot operation to compare to."""
        # Take into account that PauliRot includes a -0.5 prefactor, and mitigate it.
        qml.PauliRot(-2 * x, word, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    x = np.linspace(0.2, 1.5, 15)
    state = qml.SpecialUnitary.compute_matrix(x, 2) @ np.eye(4)[0]
    exp = np.vdot(state, qml.matrix(qml.PauliZ(0) @ qml.PauliX(1)) @ state).real

    def test_qnode_numpy(self, dev_fn):
        """Test that the QNode executes with Numpy."""
        dev = dev_fn(wires=2)
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp)

    @pytest.mark.autograd
    def test_qnode_autograd(self, dev_fn):
        """Test that the QNode executes with Autograd.
        Neither hardware-ready nor autodiff gradients are available in Autograd."""

        dev = dev_fn(wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="autograd")

        x = qml.numpy.array(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_jax(self, dev_fn, shots, atol, use_jit):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        if use_jit and shots is not None:
            pytest.skip("Hardware-ready differentiation does not support JITting yet.")
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = dev_fn(wires=2)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method, shots=shots)
        if use_jit:
            qnode = jax.jit(qnode)

        x = jax.numpy.array(self.x)
        res = qnode(x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp, atol=atol)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(x)
        assert jac.shape == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)

        # Compare to PauliRot circuits
        paulirot_qnode = qml.QNode(
            self.paulirot_comp_circuit, dev, interface="jax", diff_method=diff_method
        )
        exp_jac_fn = jax.jacobian(paulirot_qnode)
        words = qml.ops.qubit.special_unitary.pauli_basis_strings(2)
        for i, (single_x, unit_vector, word) in enumerate(zip(x, jax.numpy.eye(15), words)):
            jac = jac_fn(single_x * unit_vector)
            exp_jac = exp_jac_fn(single_x, word)
            assert qml.math.allclose(jac[i], exp_jac, atol=atol)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_torch(self, dev_fn, shots, atol):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = dev_fn(wires=2)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(
            self.circuit, dev, interface="torch", diff_method=diff_method, shots=shots
        )

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, torch.tensor(self.exp), atol=atol)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qml.math.shape(jac) == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)

        # Compare to PauliRot circuits
        paulirot_qnode = qml.QNode(
            self.paulirot_comp_circuit, dev, interface="torch", diff_method=diff_method
        )
        words = qml.ops.qubit.special_unitary.pauli_basis_strings(2)
        for i, (single_x, unit_vector, word) in enumerate(zip(x, torch.eye(15), words)):
            jac = torch.autograd.functional.jacobian(qnode, single_x * unit_vector)
            exp_jac = torch.autograd.functional.jacobian(
                partial(paulirot_qnode, word=word), single_x
            )
            assert qml.math.allclose(jac[i], exp_jac, atol=atol)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_tf(self, dev_fn, shots, atol):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = dev_fn(wires=2)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method, shots=shots)

        x = tf.Variable(self.x)
        with tf.GradientTape() as tape:
            res = qnode(x)

        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp, atol=atol)

        jac = tape.gradient(res, x)
        assert qml.math.shape(jac) == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)

        # Compare to PauliRot circuits
        paulirot_qnode = qml.QNode(
            self.paulirot_comp_circuit, dev, interface="tf", diff_method=diff_method
        )
        words = qml.ops.qubit.special_unitary.pauli_basis_strings(2)
        for i, (single_x, unit_vector, word) in enumerate(
            zip(self.x, tf.eye(15, dtype=tf.float64), words)
        ):
            x = tf.Variable(single_x * unit_vector)
            with tf.GradientTape() as tape:
                res = qnode(x)
            jac = tape.gradient(res, x)
            single_x = tf.Variable(single_x)
            with tf.GradientTape() as tape:
                res = paulirot_qnode(single_x, word)
            exp_jac = tape.gradient(res, single_x)
            assert qml.math.allclose(jac[i], exp_jac, atol=atol)


class TestTmpPauliRot:
    """Tests for the helper Operation TmpPauliRot."""

    @staticmethod
    def get_decomposition(x):
        """A nonsense function that can be trained, to convert x to a trainable value."""
        decomp = TmpPauliRot.compute_decomposition(x, [0], "X")
        return float(len(decomp))

    def test_has_matrix_false(self):
        """Test that TmpPauliRot reports to not have a matrix."""
        assert TmpPauliRot.has_matrix is False
        assert TmpPauliRot(0.2, "X", 0).has_matrix is False

    def test_has_grad_method(self):
        """Test that TmpPauliRot reports to have an analytic grad method."""
        assert TmpPauliRot.grad_method == "A"
        assert TmpPauliRot(0.2, "X", 0).grad_method == "A"

    def test_repr(self):
        """Test the string representation of TmpPauliRot."""
        rep = str(TmpPauliRot(0.2, "IX", [1, 0]))
        assert "TmpPauliRot" in rep
        assert "IX" in rep

    @pytest.mark.parametrize("word", ["X", "IZ", "YYY"])
    def test_decomposition_at_zero(self, word):
        """Test the decomposition of TmpPauliRot at zero to return an empty list."""
        wires = list(range(len(word)))
        op = TmpPauliRot(0.0, word, wires=wires)
        assert op.decomposition() == []
        assert TmpPauliRot.compute_decomposition(0.0, wires, word) == []

    @pytest.mark.autograd
    def test_decomposition_at_zero_autograd(self):
        """Test that the decomposition is a PauliRot if the theta value is trainable."""
        x = qml.numpy.array(0.0, requires_grad=True)
        with qml.queuing.AnnotatedQueue() as q:
            qml.grad(self.get_decomposition)(x)
        assert q.queue[0] == qml.PauliRot(x, "X", [0])

    @pytest.mark.jax
    def test_decomposition_at_zero_jax(self):
        """Test that the decomposition is a PauliRot if the theta value is trainable."""
        import jax

        x = jax.numpy.array(0.0)
        with qml.queuing.AnnotatedQueue() as q:
            jax.grad(self.get_decomposition)(x)
        assert _convert_op_to_numpy_data(q.queue[0]) == qml.PauliRot(0.0, "X", [0])

    @pytest.mark.tf
    def test_decomposition_at_zero_tf(self):
        """Test that the decomposition is a PauliRot if the theta value is trainable."""
        import tensorflow as tf

        x = tf.Variable(0.0)
        with qml.queuing.AnnotatedQueue() as q:
            with tf.GradientTape():
                num_ops = self.get_decomposition(x)
        assert num_ops == 1
        assert q.queue[0] == qml.PauliRot(x, "X", [0])

    @pytest.mark.torch
    def test_decomposition_at_zero_torch(self):
        """Test that the decomposition is a PauliRot if the theta value is trainable."""
        import torch

        x = torch.tensor(0.0, requires_grad=True)
        with qml.queuing.AnnotatedQueue() as q:
            num_ops = self.get_decomposition(x)
        assert num_ops == 1
        assert q.queue[0] == qml.PauliRot(x, "X", [0])

    @pytest.mark.parametrize("word", ["X", "IZ", "YYY"])
    @pytest.mark.parametrize("x", [1.2, 1e-4])
    def test_decomposition_nonzero(self, word, x):
        """Test the decomposition of TmpPauliRot away from zero to return a PauliRot."""
        wires = list(range(len(word)))
        op = TmpPauliRot(x, word, wires=wires)
        decomp = op.decomposition()
        decomp2 = TmpPauliRot.compute_decomposition(x, wires, word)
        for dec in [decomp, decomp2]:
            assert len(dec) == 1
            qml.assert_equal(dec[0], qml.PauliRot(x, word, wires))
