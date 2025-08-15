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
"""Unit tests for density matrices functions."""
# pylint: disable=import-outside-toplevel

import numpy as onp
import pytest

from pennylane import math as fn
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

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

array_funcs = [lambda x: x, onp.array, np.array, jnp.array, torch.tensor]

single_wires_list = [
    [0],
    [1],
]

multiple_wires_list = [
    [0, 1],
    [1, 0],
]
# fmt: on

c_dtypes = ["complex64", "complex128"]


def permute_two_qubit_dm(dm):
    """Permute the two qubits of a density matrix by transposing."""
    return fn.reshape(fn.transpose(fn.reshape(dm, [2] * 4), [1, 0, 3, 2]), (4, 4))


class TestDensityMatrixFromStateVectors:
    """Tests for creating a density matrix from state vectors."""

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_reduce_statevector_single_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for single wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum.reduce_statevector(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduce_statevector_full_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for full wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum.reduce_statevector(state_vector, indices=wires)
        expected = expected_density_matrix[2]
        if wires == [1, 0]:
            expected = permute_two_qubit_dm(expected)
        assert np.allclose(density_matrix, expected)

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_reduced_dm_with_state_vector_single_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the reduced_dm with state vectors for single wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.reduce_statevector(state_vector, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_state_vector_full_wires(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the reduced_dm with state vectors for full wires."""
        state_vector = array_func(state_vector)
        density_matrix = fn.reduce_statevector(state_vector, indices=wires)
        expected = expected_density_matrix[2]
        if wires == [1, 0]:
            expected = permute_two_qubit_dm(expected)
        assert np.allclose(density_matrix, expected)

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector, expected_density_matrix", state_vectors)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduce_statevector_check_state(
        self, state_vector, wires, expected_density_matrix, array_func
    ):
        """Test the density matrix from state vectors for single wires with state checking"""
        state_vector = array_func(state_vector)
        density_matrix = fn.quantum.reduce_statevector(
            state_vector, indices=wires, check_state=True
        )
        expected = expected_density_matrix[2]
        if wires == [1, 0]:
            expected = permute_two_qubit_dm(expected)

        assert np.allclose(density_matrix, expected)

    def test_state_vector_wrong_shape(self):
        """Test that wrong shaped state vector raises an error with check_state=True"""
        state_vector = [1, 0, 0]

        with pytest.raises(ValueError, match="State vector must be"):
            fn.quantum.reduce_statevector(state_vector, indices=[0], check_state=True)

    def test_state_vector_wrong_norm(self):
        """Test that state vector with wrong norm raises an error with check_state=True"""
        state_vector = [0.1, 0, 0, 0]

        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            fn.quantum.reduce_statevector(state_vector, indices=[0], check_state=True)

    def test_reduce_statevector_jax_jit(self):
        """Test jitting the density matrix from state vector function."""
        # pylint: disable=protected-access
        from jax import jit

        state_vector = jnp.array([1, 0, 0, 0])

        jitted_dens_matrix_func = jit(fn.quantum.reduce_statevector, static_argnums=[1, 2])

        density_matrix = jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)
        assert np.allclose(density_matrix, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_wrong_shape_jax_jit(self):
        """Test jitting the density matrix from state vector with wrong shape."""
        # pylint: disable=protected-access
        from jax import jit

        state_vector = jnp.array([1, 0, 0])

        jitted_dens_matrix_func = jit(fn.quantum.reduce_statevector, static_argnums=[1, 2])

        with pytest.raises(ValueError, match="State vector must be"):
            jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)

    @pytest.mark.parametrize("c_dtype", c_dtypes)
    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("state_vector", list(zip(*state_vectors))[0])
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_density_matrix_c_dtype(self, array_func, state_vector, wires, c_dtype):
        """Test different complex dtype."""
        state_vector = array_func(state_vector)
        if fn.get_interface(state_vector) == "jax" and c_dtype == "complex128":
            pytest.skip("Jax does not support complex 128")
        density_matrix = fn.reduce_statevector(state_vector, indices=wires, c_dtype=c_dtype)
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
        density_matrix = fn.reduce_dm(density_matrix, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrix[wires[0]])

    @pytest.mark.parametrize("density_matrix", list(zip(*density_matrices))[0])
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_matrix_full_wires(self, density_matrix, wires):
        """Test the reduced_dm with matrix for full wires."""
        returned_density_matrix = fn.reduce_dm(density_matrix, indices=wires)
        expected = density_matrix
        if wires == [1, 0]:
            expected = permute_two_qubit_dm(expected)
        assert np.allclose(returned_density_matrix, expected)

    @pytest.mark.parametrize("density_matrix", list(zip(*density_matrices))[0])
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduce_dm_check(self, density_matrix, wires):
        """Test the density matrix from matrices for single wires with state checking"""
        # pylint: disable=protected-access
        returned_density_matrix = fn.quantum.reduce_dm(
            density_matrix, indices=wires, check_state=True
        )
        expected = density_matrix
        if wires == [1, 0]:
            expected = permute_two_qubit_dm(expected)

        assert np.allclose(returned_density_matrix, expected)

    def test_matrix_wrong_shape(self):
        """Test that wrong shaped state vector raises an error with check_state=True"""
        density_matrix = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            fn.quantum.reduce_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_wrong_trace(self):
        """Test that density matrix with wrong trace raises an error with check_state=True"""
        density_matrix = [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        with pytest.raises(ValueError, match="The trace of the density matrix should be one."):
            fn.quantum.reduce_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_not_hermitian(self):
        """Test that non hermitian matrix raises an error with check_state=True"""
        density_matrix = [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, 0.9]]

        with pytest.raises(ValueError, match="The matrix is not Hermitian."):
            fn.quantum.reduce_dm(density_matrix, indices=[0], check_state=True)

    def test_matrix_not_positive_definite(self):
        """Test that non hermitian matrix raises an error with check_state=True"""
        density_matrix = [[3, 0], [0, -2]]

        with pytest.raises(ValueError, match="The matrix is not positive semi-definite."):
            fn.quantum.reduce_dm(density_matrix, indices=[0], check_state=True)

    def test_reduce_statevector_jax_jit(self):
        """Test jitting the density matrix from state vector function."""
        from jax import jit

        state_vector = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        jitted_dens_matrix_func = jit(fn.quantum.reduce_dm, static_argnums=[1, 2])

        density_matrix = jitted_dens_matrix_func(state_vector, indices=(0,), check_state=True)
        assert np.allclose(density_matrix, [[1, 0], [0, 0]])

    def test_wrong_shape_jax_jit(self):
        """Test jitting the density matrix from state vector with wrong shape."""
        # pylint: disable=protected-access
        from jax import jit

        state_vector = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        jitted_dens_matrix_func = jit(fn.quantum.reduce_dm, static_argnums=[1, 2])

        with pytest.raises(ValueError, match="Density matrix must be of shape"):
            jitted_dens_matrix_func(state_vector, indices=(0, 1), check_state=True)

    @pytest.mark.parametrize("c_dtype", c_dtypes)
    @pytest.mark.parametrize("density_matrix", list(zip(*density_matrices))[0])
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_density_matrix_c_dtype(self, density_matrix, wires, c_dtype):
        """Test different complex dtype."""
        if fn.get_interface(density_matrix) == "jax" and c_dtype == "complex128":
            pytest.skip("Jax does not support complex 128")
        density_matrix = fn.reduce_dm(density_matrix, indices=wires, c_dtype=c_dtype)
        if fn.get_interface(density_matrix) == "torch":
            if c_dtype == "complex64":
                c_dtype = torch.complex64
            elif c_dtype == "complex128":
                c_dtype = torch.complex128
        assert density_matrix.dtype == c_dtype


class TestDensityMatrixBroadcasting:
    """Test that broadcasting works as expected for the reduced_dm functions"""

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_sv_broadcast_single_wires(self, wires, array_func):
        """Test that broadcasting works for state vectors and single wires"""
        state_vector = array_func([sv[0] for sv in state_vectors])
        expected_density_matrices = array_func([sv[1][wires[0]] for sv in state_vectors])

        _density_matrices = fn.reduce_statevector(state_vector, indices=wires)
        assert np.allclose(_density_matrices, expected_density_matrices)

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_state_vector_full_wires(self, wires, array_func):
        """Test that broadcasting works for state vectors and full wires"""
        state_vector = array_func([sv[0] for sv in state_vectors])

        if wires == [0, 1]:
            expected_density_matrices = array_func([sv[1][2] for sv in state_vectors])
        else:
            expected_density_matrices = array_func(
                [permute_two_qubit_dm(sv[1][2]) for sv in state_vectors]
            )

        _density_matrices = fn.reduce_statevector(state_vector, indices=wires)
        assert np.allclose(_density_matrices, expected_density_matrices)

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("wires", single_wires_list)
    def test_reduced_dm_with_matrix_single_wires(self, wires, array_func):
        """Test that broadcasting works for density matrices and single wires"""
        density_matrix = array_func([dm[0] for dm in density_matrices[:4]])
        expected_density_matrices = array_func([dm[1][wires[0]] for dm in density_matrices[:4]])

        density_matrix = fn.reduce_dm(density_matrix, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrices)

    @pytest.mark.parametrize("array_func", array_funcs)
    @pytest.mark.parametrize("wires", multiple_wires_list)
    def test_reduced_dm_with_matrix_full_wires(self, wires, array_func):
        """Test that broadcasting works for density matrices and full wires"""
        density_matrix = array_func([dm[0] for dm in density_matrices[:4]])

        if wires == [0, 1]:
            expected_density_matrices = density_matrix
        else:
            expected_density_matrices = array_func(
                [permute_two_qubit_dm(dm[0]) for dm in density_matrices[:4]]
            )

        density_matrix = fn.reduce_dm(density_matrix, indices=wires)
        assert np.allclose(density_matrix, expected_density_matrices)
