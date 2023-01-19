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

import pytest

import numpy as np

import pennylane as qml
from pennylane.ops.qubit.matrix_ops import pauli_basis
from pennylane.transforms.insert_paulirot import (
    get_one_parameter_generators,
    get_one_parameter_coeffs,
)


@pytest.mark.parametrize("n", [1, 2, 3])
class TestGetOneParameterGenerators:
    """Tests for the effective generators computing function
    get_one_parameter_generators."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_Omegas_jax(self, n, use_jit):
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

    @pytest.mark.tf
    def test_Omegas_tf(self, n):
        """Test that generators are computed correctly in Tensorflow."""
        import tensorflow as tf

        np.random.seed(14521)
        d = 4**n - 1
        theta = tf.Variable(np.random.random(d))
        Omegas = get_one_parameter_generators(theta, n, "tf")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    @pytest.mark.torch
    def test_Omegas_torch(self, n):
        """Test that generators are computed correctly in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "torch")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_Omegas_autograd(self, n):
        """Test that generators are computed correctly in Autograd."""
        np.random.seed(14521)
        d = 4**n - 1
        theta = qml.numpy.array(np.random.random(d), requires_grad=True)
        Omegas = get_one_parameter_generators(theta, n, "autograd")
        assert Omegas.shape == (d, 2**n, 2**n)
        assert all(qml.math.allclose(qml.math.conj(qml.math.T(O)), -O) for O in Omegas)


@pytest.mark.parametrize("n", [1, 2])
class TestGetOneParameterGeneratorsDiffability:
    """Tests for the effective generators computing function
    get_one_parameter_generators to be differentiable."""

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_Omegas_jacobian_jax(self, n, use_jit):
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
    def test_Omegas_jacobian_tf(self, n):
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
    def test_Omegas_jacobian_torch(self, n):
        """Test that generators are differentiable in Torch."""
        import torch

        np.random.seed(14521)
        d = 4**n - 1
        theta = torch.tensor(np.random.random(d), requires_grad=True)
        fn = lambda theta: get_one_parameter_generators(theta, n, "torch")
        dOmegas = torch.autograd.functional.jacobian(fn, theta)
        assert dOmegas.shape == (d, 2**n, 2**n, d)

    # Autograd does not support differentiating expm.
    @pytest.mark.xfail
    @pytest.mark.autograd
    def test_Omegas_jacobian_autograd(self, n):
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
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_omegas_jax(self, n, use_jit):
        """Test that the coefficients of the generators are computed correctly in JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jax import numpy as jnp

        np.random.seed(14521)
        d = 4**n - 1
        theta = jnp.array(np.random.random(d))
        fn = (
            jax.jit(get_one_parameter_coeffs, static_argnums=[1, 2])
            if use_jit
            else get_one_parameter_coeffs
        )
        omegas = fn(theta, n, "jax")
        assert omegas.shape == (d, d)
        assert jnp.allclose(omegas.real, 0)

        basis = pauli_basis(n)
        reconstructed_Omegas = jnp.tensordot(omegas, basis, axes=[[0], [0]])
        Omegas = get_one_parameter_generators(theta, n, "jax")
        assert jnp.allclose(reconstructed_Omegas, Omegas)

    @pytest.mark.tf
    def test_omegas_tf(self, n):
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
    def test_omegas_torch(self, n):
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
    def test_omegas_autograd(self, n):
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


class TestInsertPauliRot:
    """Tests for the qfunc_transform insert_paulirot."""

    WIP
