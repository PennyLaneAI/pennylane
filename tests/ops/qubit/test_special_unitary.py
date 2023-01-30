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
    get_one_parameter_coeffs,
    get_one_parameter_generators,
    special_unitary_matrix,
    TmpPauliRot,
    _detach,
    _pauli_letters,
    _pauli_matrices,
)
from pennylane.wires import Wires


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


class TestDetach:
    """Test the utility function _detach."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jax(self, use_jit):
        """Test that _detach works with JAX."""
        import jax

        x = jax.numpy.array(0.3)
        fn = jax.jit(_detach, static_argnums=1) if use_jit else _detach
        jac = jax.jacobian(fn, argnums=0)(x, "jax")
        assert jax.numpy.isclose(jac, 0.0)

    @pytest.mark.torch
    def test_torch(self):
        """Test that _detach works with Torch."""
        import torch

        x = torch.tensor(0.3, requires_grad=True)
        assert x.requires_grad is True
        detached_x = _detach(x, "torch")
        assert detached_x.requires_grad is False
        jac = torch.autograd.functional.jacobian(partial(_detach, interface="torch"), x)
        assert qml.math.isclose(jac, jac * 0.0)

    @pytest.mark.tf
    def test_tf(self):
        """Test that _detach works with Tensorflow."""
        import tensorflow as tf

        x = tf.Variable(0.3)
        assert x.trainable is True
        detached_x = _detach(x, "tf")
        assert not hasattr(detached_x, "trainable")
        with tf.GradientTape() as t:
            out = _detach(x, "tf")
        jac = t.jacobian(out, x)
        assert jac is None


class TestGetOneParameterGenerators:
    """Tests for the effective generators computing function
    get_one_parameter_generators."""

    @pytest.mark.jax
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jax(self, n, use_jit):
        """Test that generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        fn = (
            jax.jit(get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_generators
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
            jax.jit(get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_generators
        )
        basis = pauli_basis(n)
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

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        Omegas = get_one_parameter_generators(theta, n, "tf")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.tf
    def test_tf_pauli_generated(self):
        """Test that generators match Pauli words."""
        import tensorflow as tf

        n = 1
        basis = pauli_basis(n)
        d = 4**n - 1
        for i, (theta, pauli_mat) in enumerate(zip(np.eye(d), basis)):
            theta = tf.Variable(theta)
            Omegas = get_one_parameter_generators(theta, n, "tf")
            assert Omegas.shape == (d, 2**n, 2**n)
            assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)
            assert qml.math.allclose(Omegas[i], 1j * pauli_mat)

    @pytest.mark.torch
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_torch(self, n):
        """Test that generators are computed correctly in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "torch")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.torch
    def test_torch_pauli_generated(self):
        """Test that generators match Pauli words."""
        import torch

        n = 1
        basis = pauli_basis(n)
        d = 4**n - 1
        for i, (theta, pauli_mat) in enumerate(zip(torch.eye(d, requires_grad=True), basis)):
            Omegas = get_one_parameter_generators(theta, n, "torch")
            assert Omegas.shape == (d, 2**n, 2**n)
            assert all(
                qml.math.allclose(qml.math.conj(qml.math.T(O)), -O, atol=3e-8) for O in Omegas
            )
            assert qml.math.allclose(Omegas[i], 1j * pauli_mat)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_autograd(self, n):
        """Test that generators are computed correctly in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "autograd")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    def test_raises_autograd(self):
        """Test that computing generators raises an error when attempting to use Autograd."""
        with pytest.raises(NotImplementedError, match="expm is not differentiable in Autograd"):
            get_one_parameter_generators(None, None, "autograd")

    def test_raises_unknown_interface(self):
        """Test that computing generators raises an error when attempting
        to use an unknown interface."""
        with pytest.raises(ValueError, match="The interface test is not supported"):
            get_one_parameter_generators(None, None, "test")

    def test_raises_broadcasting(self):
        """Test that computing generators raises an error when attempting
        to use broadcasting."""
        theta = np.random.random((2, 3))
        with pytest.raises(ValueError, match="Broadcasting is not supported"):
            get_one_parameter_generators(theta, 1, "dummy")


@pytest.mark.parametrize("n", [1, 2])
class TestGetOneParameterGeneratorsDiffability:
    """Tests for the effective generators computing function
    get_one_parameter_generators to be differentiable."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_jacobian_jax(self, n, use_jit):
        """Test that generators are differentiable in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d), dtype=jnp.complex128)
        fn = (
            jax.jit(get_one_parameter_generators, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_generators
        )
        dOmegas = jax.jacobian(fn, holomorphic=True)(theta, n, "jax")
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.tf
    def test_jacobian_tf(self, n):
        """Test that generators are differentiable in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        with tf.GradientTape() as t:
            Omegas = get_one_parameter_generators(theta, n, "tf")
        dOmegas = t.jacobian(Omegas, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    @pytest.mark.torch
    def test_jacobian_torch(self, n):
        """Test that generators are differentiable in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)

        def fn(theta):
            return get_one_parameter_generators(theta, n, "torch")

        dOmegas = torch.autograd.functional.jacobian(fn, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_jacobian_autograd(self, n):
        """Test that generators are differentiable in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        dOmegas = qml.jacobian(get_one_parameter_generators)(theta, n, "autograd")
        assert dOmegas.shape == (d, 2**n, 2**n, d)


@pytest.mark.parametrize("n", [1, 2, 3])
class TestGetOneParameterCoeffs:
    """Tests for the coefficients of effective generators computing function
    get_one_parameter_coeffs."""

    @pytest.mark.jax
    def test_jax(self, n):
        """Test that the coefficients of the generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        omegas = get_one_parameter_coeffs(theta, n, "jax")
        assert omegas.shape == (d, d)
        assert jnp.allclose(omegas.real, 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = jnp.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "jax")
        assert jnp.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.tf
    def test_tf(self, n):
        """Test that the coefficients of the generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        omegas = get_one_parameter_coeffs(theta, n, "tf")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "tf")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.torch
    def test_torch(self, n):
        """Test that the coefficients of the generators are computed correctly in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        omegas = get_one_parameter_coeffs(theta, n, "torch")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "torch")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_autograd(self, n):
        """Test that the coefficients of the generators are computed correctly in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        omegas = get_one_parameter_coeffs(theta, n, "autograd")
        assert omegas.shape == (d, d)
        assert qml.math.allclose(qml.math.real(omegas), 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = qml.math.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "autograd")
        assert qml.math.allclose(reconstructed_Omegas, Omegas)


theta_1 = np.array([0.4, 0.1, 0.1])
theta_2 = np.array([0.4, 0.1, 0.1, 0.6, 0.2, 0.3, 0.1, 0.2, 0, 0.2, 0.2, 0.2, 0.1, 0.5, 0.2])
n_and_theta = [(1, theta_1), (2, theta_2)]


class TestTmpPauliRot:
    """Tests for the helper Operation TmpPauliRot."""

    def test_has_matrix_false(self):
        """Test that TmpPauliRot reports to not have a matrix."""
        assert not TmpPauliRot.has_matrix
        assert not TmpPauliRot(0.2, "X", 0).has_matrix

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
            assert qml.equal(dec[0], qml.PauliRot(x, word, wires))


class TestSpecialUnitary:
    """Tests for the Operation ``SpecialUnitary``."""

    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_numpy(self, n, theta):
        """Test that SpecialUnitary falls back to QubitUnitary with NumPy inputs."""

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
    def test_decomposition_broadcasted_numpy(self, n, theta):
        """Test that the broadcasted SpecialUnitary falls back to QubitUnitary with NumPy inputs."""
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

    @pytest.mark.jax
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_jax(self, n, theta):
        """Test that a trainable SpecialUnitary in JAX
        decomposes into a non-trainable SpecialUnitary and TmpPauliRot ops."""
        import jax

        jax.config.update("jax_enable_x64", True)

        d = 4**n - 1
        words = pauli_words(n)
        wires = list(range(n))

        def assertion_fn(theta):
            """Wrapper function to allow marking the parameters as trainable in JAX."""
            decomp = qml.SpecialUnitary.compute_decomposition(theta, wires, n)
            decomp2 = qml.SpecialUnitary(theta, wires).decomposition()
            for dec in [decomp, decomp2]:
                assert len(dec) == d + 1
                for w, op in zip(words, dec[:-1]):
                    assert qml.equal(
                        TmpPauliRot(0.0, w, wires=wires),
                        op,
                        check_trainability=False,
                        check_interface=False,
                    )
                assert qml.equal(qml.SpecialUnitary(_detach(theta, "jax"), wires=wires), dec[-1])

            decomp = qml.SpecialUnitary.compute_decomposition(_detach(theta, "jax"), wires, n)
            decomp2 = qml.SpecialUnitary(_detach(theta, "jax"), wires).decomposition()
            mat = special_unitary_matrix(_detach(theta, "jax"), n)
            for dec in [decomp, decomp2]:
                assert len(dec) == 1
                assert qml.equal(qml.QubitUnitary(mat, wires=wires), dec[0])

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
        words = pauli_words(n)
        wires = list(range(n))
        theta = torch.tensor(theta, requires_grad=True)
        decomp = qml.SpecialUnitary.compute_decomposition(theta, wires, n)
        decomp2 = qml.SpecialUnitary(theta, wires).decomposition()
        for dec in [decomp, decomp2]:
            assert len(dec) == d + 1
            for w, op in zip(words, dec[:-1]):
                assert qml.equal(
                    TmpPauliRot(0.0, w, wires=wires),
                    op,
                    check_trainability=False,
                    check_interface=False,
                )
            assert qml.equal(qml.SpecialUnitary(_detach(theta, "torch"), wires=wires), dec[-1])

        decomp = qml.SpecialUnitary.compute_decomposition(_detach(theta, "torch"), wires, n)
        decomp2 = qml.SpecialUnitary(_detach(theta, "torch"), wires).decomposition()
        mat = special_unitary_matrix(_detach(theta, "torch"), n)
        for dec in [decomp, decomp2]:
            assert len(dec) == 1
            assert qml.equal(qml.QubitUnitary(mat, wires=wires), dec[0])

    @pytest.mark.tf
    @pytest.mark.parametrize("n, theta", n_and_theta)
    def test_decomposition_tf(self, n, theta):
        """Test that a trainable SpecialUnitary in Tensorflow
        decomposes into a non-trainable SpecialUnitary and TmpPauliRot ops."""
        import tensorflow as tf

        d = 4**n - 1
        words = pauli_words(n)
        wires = list(range(n))
        theta = tf.Variable(theta)
        with tf.GradientTape():
            decomp = qml.SpecialUnitary.compute_decomposition(theta, wires, n)
            decomp2 = qml.SpecialUnitary(theta, wires).decomposition()
            for dec in [decomp, decomp2]:
                assert len(dec) == d + 1
                for w, op in zip(words, dec[:-1]):
                    assert qml.equal(
                        TmpPauliRot(0.0, w, wires=wires),
                        op,
                        check_trainability=False,
                        check_interface=False,
                    )
                assert qml.equal(qml.SpecialUnitary(_detach(theta, "tf"), wires=wires), dec[-1])

            decomp = qml.SpecialUnitary.compute_decomposition(_detach(theta, "tf"), wires, n)
            decomp2 = qml.SpecialUnitary(_detach(theta, "tf"), wires).decomposition()
            mat = special_unitary_matrix(_detach(theta, "tf"), n)
            for dec in [decomp, decomp2]:
                assert len(dec) == 1
                assert qml.equal(qml.QubitUnitary(mat, wires=wires), dec[0])

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


class TestSpecialUnitaryIntegration:
    """Test that the operation SpecialUnitary is executable and
    differentiable in a QNode context, both with automatic differentiation
    and parameter-shift rules."""

    @staticmethod
    def circuit(x):
        """Test circuit involving a single SpecialUnitary operations."""
        qml.SpecialUnitary(x, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    x = np.linspace(0.2, 1.5, 15)
    state = special_unitary_matrix(x, 2) @ np.eye(4)[0]
    exp = np.vdot(state, qml.matrix(qml.PauliZ(0) @ qml.PauliX(1)) @ state).real

    def test_qnode_numpy(self):
        """Test that the QNode executes with Numpy."""
        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface=None)

        res = qnode(self.x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp)

    def test_qnode_autograd(self):
        """Test that the QNode executes with Autograd.
        Neither hardware-ready nor autodiff gradients are available in Autograd."""

        dev = qml.device("default.qubit", wires=2)
        qnode = qml.QNode(self.circuit, dev, interface="autograd")

        x = qml.numpy.array(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp)

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_jax(self, shots, atol, use_jit):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        if use_jit and shots is not None:
            pytest.skip("Hardware-ready differentiation does not support JITting yet.")
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2, shots=shots)
        qnode = qml.QNode(self.circuit, dev, interface="jax")
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
        assert qml.math.shape(jac) == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)

    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_torch(self, shots, atol):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = qml.device("default.qubit", wires=2, shots=shots)
        qnode = qml.QNode(self.circuit, dev, interface="torch")

        x = torch.tensor(self.x, requires_grad=True)
        res = qnode(x)
        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, torch.tensor(self.exp), atol=atol)

        jac = torch.autograd.functional.jacobian(qnode, x)
        assert qml.math.shape(jac) == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)

    @pytest.mark.parametrize("shots, atol", [(None, 1e-6), (10000, 1e-1)])
    def test_qnode_tf(self, shots, atol):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2, shots=shots)
        qnode = qml.QNode(self.circuit, dev, interface="tf")

        x = tf.Variable(self.x)
        with tf.GradientTape() as tape:
            res = qnode(x)

        assert qml.math.shape(res) == ()
        assert qml.math.isclose(res, self.exp, atol=atol)

        jac = tape.jacobian(res, x)
        assert qml.math.shape(jac) == (15,)
        assert not qml.math.allclose(jac, jac * 0.0)
