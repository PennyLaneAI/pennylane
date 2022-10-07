# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for density matrices functions.
"""

import numpy as onp
import pytest

from pennylane import numpy as np
from pennylane import math as fn

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

state_00 = [1, 0, 0, 0]
state_01 = [0, 1, 0, 0]
state_10 = [0, 0, 1, 0]
state_11 = [0, 0, 0, 1]

state_00_10 = [1, 0, 1, 0] / onp.sqrt(2)
state_01_11 = [0, 1, 0, 1] / onp.sqrt(2)

mat_00 = onp.zeros((4, 4))
mat_00[0, 0] = 1

mat_01 = onp.zeros((4, 4))
mat_01[1, 1] = 1

mat_10 = onp.zeros((4, 4))
mat_10[2, 2] = 1

mat_11 = onp.zeros((4, 4))
mat_11[3, 3] = 1

mat_0 = onp.zeros((2, 2))
mat_0[0, 0] = 1

mat_1 = onp.zeros((2, 2))
mat_1[1, 1] = 1

mat_00_10 = onp.zeros((4, 4))
mat_00_10[0, 0] = 0.5
mat_00_10[2, 2] = 0.5
mat_00_10[0, 2] = 0.5
mat_00_10[2, 0] = 0.5

mat_01_11 = onp.zeros((4, 4))
mat_01_11[1, 1] = 0.5
mat_01_11[3, 3] = 0.5
mat_01_11[1, 3] = 0.5
mat_01_11[3, 1] = 0.5

mat_0_1 = [[0.5, 0.5], [0.5, 0.5]]

# fmt: off
state_vectors = [
    (state_00, (mat_0, mat_0, mat_00)),
    (state_01, (mat_0, mat_1, mat_01)),
    (state_10, (mat_1, mat_0, mat_10)),
    (state_11, (mat_1, mat_1, mat_11)),
    (state_00_10, (mat_0_1, mat_0, mat_00_10)),
    (state_01_11, (mat_0_1, mat_1, mat_01_11))]

array_funcs = [lambda x: x, onp.array, np.array, jnp.array, torch.tensor, tf.Variable, tf.constant]

single_wires_list = [
    [0],
    [1],
]

multiple_wires_list = [
    [0, 1]
]
# fmt: on

c_dtypes = ["complex64", "complex128"]


class TestDensityMatrixFromStateVectors:
    """Tests for creating a density matrix from state vectors."""

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_density_matrix_from_state_vector_single_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for single wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum._density_matrix_from_state_vector(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_density_matrix_from_state_vector_full_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for full wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum._density_matrix_from_state_vector(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[2])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_reduced_dm_with_state_vector_single_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the reduced_dm with state vectors for single wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.reduced_dm(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_state_vector_full_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the reduced_dm with state vectors for full wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.reduced_dm(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[2])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_density_matrix_from_state_vector_check_state(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for single wires with state checking"""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum.reduced_dm(state_vector, indices=wires, check_state=True)

        assert np.allclose(density_matrix, expected_density_matrix[2])

    def test_state_vector_wrong_shape(self):
        """Test that wrong shaped state vector raises an error with check_state=True"""
        state_vector = [1, 0, 0]

        with pytest.raises(ValueError, match="State vector must be"):
            fn.quantum.reduced_dm(state_vector, indices=[0], check_state=True)

    def test_state_vector_wrong_norm(self):
        """Test that state vector with wrong norm raises an error with check_state=True"""
        state_vector = [0.1, 0, 0, 0]

        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            fn.quantum.reduced_dm(state_vector, indices=[0], check_state=True)

    def test_density_matrix_from_state_vector_jax_jit(self):
        """Test jitting the density matrix from state vector function."""
        from jax import jit
        import jax.numpy as jnp

        state_vector = jnp.array([1, 0, 0, 0])

        jitted_dens_matrix_func = jit(
            fn.quantum._density_matrix_from_state_vector, static_argnums=[1, 2]
        )

        density_matrix = jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)
        assert np.allclose(density_matrix, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_wrong_shape_jax_jit(self):
        """Test jitting the density matrix from state vector with wrong shape."""
        from jax import jit
        import jax.numpy as jnp

        state_vector = jnp.array([1, 0, 0])

        jitted_dens_matrix_func = jit(
            fn.quantum._density_matrix_from_state_vector, static_argnums=[1, 2]
        )

        with pytest.raises(ValueError, match="State vector must be"):
            jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)

    def test_density_matrix_tf_jit(self):
        """Test jitting the density matrix from state vector function with Tf."""
        import tensorflow as tf
        from functools import partial

        state_vector = tf.Variable([1, 0, 0, 0], dtype=tf.complex128)

        density_matrix = partial(fn.reduced_dm, indices=[0])

        density_matrix = tf.function(
            density_matrix,
            jit_compile=True,
            input_signature=(tf.TensorSpec(shape=(4,), dtype=tf.complex128),),
        )
        density_matrix = density_matrix(state_vector)
        assert np.allclose(density_matrix, [[1, 0], [0, 0]])

    @pytest.mark.parametrize("c_dtype", c_dtypes)
    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_density_matrix_c_dtype(
        self, array_func, state_vector, wires, c_dtype, expected_density_matrix
    ):
        """Test different complex dtype."""
        state_vector = array_func(state_vector)
        if fn.get_interface(state_vector) == "jax" and c_dtype == "complex128":
            pytest.skip("Jax does not support complex 128")
        density_matrix = fn.reduced_dm(state_vector, indices=wires, c_dtype=c_dtype)
        if fn.get_interface(state_vector) == "torch":
            if c_dtype == "complex64":
                c_dtype = torch.complex64
            elif c_dtype == "complex128":
                c_dtype = torch.complex128
        assert density_matrix.dtype == c_dtype


# fmt: off
density_matrices = [
    (mat_00, (mat_0, mat_0)),
    (mat_01, (mat_0, mat_1)),
    (mat_10, (mat_1, mat_0)),
    (mat_11, (mat_1, mat_1)),
    (onp.array(mat_00), (mat_0, mat_0)),
    (onp.array(mat_01), (mat_0, mat_1)),
    (onp.array(mat_10), (mat_1, mat_0)),
    (onp.array(mat_11), (mat_1, mat_1)),
    (np.array(mat_00), (mat_0, mat_0)),
    (np.array(mat_01), (mat_0, mat_1)),
    (np.array(mat_10), (mat_1, mat_0)),
    (np.array(mat_11), (mat_1, mat_1)),
    (jnp.array(mat_00), (mat_0, mat_0)),
    (jnp.array(mat_01), (mat_0, mat_1)),
    (jnp.array(mat_10), (mat_1, mat_0)),
    (jnp.array(mat_11), (mat_1, mat_1)),
    (torch.tensor(mat_00), (mat_0, mat_0)),
    (torch.tensor(mat_01), (mat_0, mat_1)),
    (torch.tensor(mat_10), (mat_1, mat_0)),
    (torch.tensor(mat_11), (mat_1, mat_1)),
    (tf.Variable(mat_00), (mat_0, mat_0)),
    (tf.Variable(mat_01), (mat_0, mat_1)),
    (tf.Variable(mat_10), (mat_1, mat_0)),
    (tf.Variable(mat_11), (mat_1, mat_1)),
    (tf.constant(mat_00), (mat_0, mat_0)),
    (tf.constant(mat_01), (mat_0, mat_1)),
    (tf.constant(mat_10), (mat_1, mat_0)),
    (tf.constant(mat_11), (mat_1, mat_1)),
    (mat_00_10, (mat_0_1, mat_0)),
    (mat_01_11, (mat_0_1, mat_1)),
]

# fmt: on


class TestDensityMatrixFromMatrix:
    """Tests for the (reduced) density matrix for matrix."""

    @pytest.mark.parametrize("density_matrix, expected_density_matrix", density_matrices)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_reduced_dm_with_matrix_single_wires(
        self, density_matrix, wires, expected_density_matrix
    ):
        """Test the reduced_dm with matrix for single wires."""
        density_matrix = fn.reduced_dm(density_matrix, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("density_matrix, expected_density_matrix", density_matrices)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_matrix_full_wires(
        self, density_matrix, wires, expected_density_matrix
    ):
        """Test the reduced_dm with matrix for full wires."""
        returned_density_matrix = fn.reduced_dm(density_matrix, indices=wires)

        assert np.allclose(density_matrix, returned_density_matrix)

    @pytest.mark.parametrize("density_matrix, expected_density_matrix", density_matrices)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_density_matrix_from_matrix_check(self, density_matrix, wires, expected_density_matrix):
        """Test the density matrix from matrices for single wires with state checking"""
        returned_density_matrix = fn.quantum._density_matrix_from_matrix(
            density_matrix, indices=wires, check_state=True
        )
        assert np.allclose(density_matrix, returned_density_matrix)

    def test_matrix_wrong_shape(self):
        """Test that wrong shaped state vector raises an error with check_state=True"""
        density_matrix = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            fn.quantum.reduced_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_wrong_trace(self):
        """Test that density matrix with wrong trace raises an error with check_state=True"""
        density_matrix = [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        with pytest.raises(ValueError, match="The trace of the density matrix should be one."):
            fn.quantum.reduced_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_not_hermitian(self):
        """Test that non hermitian matrix raises an error with check_state=True"""
        density_matrix = [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.9]]

        with pytest.raises(ValueError, match="The matrix is not Hermitian."):
            fn.quantum.reduced_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_not_positive_definite(self):
        """Test that non hermitian matrix raises an error with check_state=True"""
        density_matrix = [[3, 0], [0, -2]]

        with pytest.raises(ValueError, match="The matrix is not positive semi-definite."):
            fn.quantum.reduced_dm(density_matrix, indices=[0], check_state=True)

    def test_density_matrix_from_state_vector_jax_jit(self):
        """Test jitting the density matrix from state vector function."""
        from jax import jit
        import jax.numpy as jnp

        state_vector = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        jitted_dens_matrix_func = jit(fn.quantum.reduced_dm, static_argnums=[1, 2])

        density_matrix = jitted_dens_matrix_func(state_vector, indices=(0,), check_state=True)
        assert np.allclose(density_matrix, [[1, 0], [0, 0]])

    def test_wrong_shape_jax_jit(self):
        """Test jitting the density matrix from state vector with wrong shape."""
        from jax import jit
        import jax.numpy as jnp

        state_vector = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        jitted_dens_matrix_func = jit(fn.quantum._density_matrix_from_matrix, static_argnums=[1, 2])

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)

    def test_density_matrix_tf_jit(self):
        """Test jitting the density matrix from density matrix function with Tf."""
        import tensorflow as tf
        from functools import partial

        d_mat = tf.Variable(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=tf.complex128,
        )
        density_matrix = partial(fn.reduced_dm, indices=[0])

        density_matrix = tf.function(
            density_matrix,
            jit_compile=True,
            input_signature=(tf.TensorSpec(shape=(4, 4), dtype=tf.complex128),),
        )
        density_matrix = density_matrix(d_mat)
        assert np.allclose(density_matrix, [[1, 0], [0, 0]])

    @pytest.mark.parametrize("c_dtype", c_dtypes)
    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("density_matrix, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_density_matrix_c_dtype(
        self, array_func, density_matrix, wires, c_dtype, expected_density_matrix
    ):
        """Test different complex dtype."""
        if fn.get_interface(density_matrix) == "jax" and c_dtype == "complex128":
            pytest.skip("Jax does not support complex 128")
        density_matrix = fn.reduced_dm(density_matrix, indices=wires, c_dtype=c_dtype)
        if fn.get_interface(density_matrix) == "torch":
            if c_dtype == "complex64":
                c_dtype = torch.complex64
            elif c_dtype == "complex128":
                c_dtype = torch.complex128
        assert density_matrix.dtype == c_dtype
