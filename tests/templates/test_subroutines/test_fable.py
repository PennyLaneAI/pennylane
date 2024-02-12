# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for FABLE.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.subroutines.fable import FABLE


class TestFable:
    """Tests for the FABLE template."""

    @pytest.fixture
    def input_matrix(self):
        """Common input matrix used in multiple tests."""
        return np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                [0.82429855, 0.82429855, 0.98175843, 0.98175843],
                [0.99675093, 0.99675093, 0.83514837, 0.83514837],
            ]
        )

    def test_standard_validity(self, input_matrix):
        """Check the operation using the assert_valid function."""
        op = FABLE(input_matrix, tol=0.01)
        qml.ops.functions.assert_valid(op)

    # pylint: disable=protected-access
    def test_flatten_unflatten(self, input_matrix):
        """Test the flatten and unflatten methods."""
        op = FABLE(input_matrix, tol=0.01)
        data, metadata = op._flatten()
        assert data is op.data
        assert metadata == 0.01
        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)

    def test_fable_real(self, input_matrix):
        ancilla = ["ancilla"]
        s = int(np.log2(np.array(input_matrix).shape[0]))
        wires_i = [f"i{index}" for index in range(s)]
        wires_j = [f"j{index}" for index in range(s)]
        wire_order = ancilla + wires_i[::-1] + wires_j[::-1]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            FABLE(input_matrix, tol=0.01)
            return qml.state()

        expected = (
            len(input_matrix)
            * qml.matrix(circuit, wire_order=wire_order)().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )
        assert np.allclose(input_matrix, expected)

    @pytest.mark.jax
    def test_fable_jax(self, input_matrix):
        """Test that the Fable operator matrix is correct for jax."""
        import jax.numpy as jnp

        ancilla = ["ancilla"]
        s = int(np.log2(np.array(input_matrix).shape[0]))
        wires_i = [f"i{index}" for index in range(s)]
        wires_j = [f"j{index}" for index in range(s)]
        wire_order = ancilla + wires_i[::-1] + wires_j[::-1]

        jax_matrix = jnp.array(input_matrix)
        op = FABLE(jax_matrix, 0)

        M = (
            len(jax_matrix)
            * qml.matrix(op, wire_order=wire_order).real[0 : len(jax_matrix), 0 : len(jax_matrix)]
        )
        assert np.allclose(M, jax_matrix)
        assert qml.math.get_interface(M) == "jax"

    def test_fable_imaginary_error(self, input_matrix):
        """Test if a ValueError is raised when imaginary values are passed in."""
        imaginary_matrix = input_matrix.astype(np.complex128)
        imaginary_matrix[0, 0] += 0.00001j  # Add a small imaginary component

        with pytest.raises(
            ValueError, match="Support for imaginary values has not been implemented."
        ):
            FABLE(imaginary_matrix, tol=0.01)

    def test_fable_normalization_error(self, input_matrix):
        """Test if a ValueError is raised when the normalization factor is greater than 1."""
        input_matrix[0, 0] += 10

        with pytest.raises(ValueError, match="The subnormalization factor should be lower than 1."):
            FABLE(input_matrix, tol=0.01)

    def test_warning_for_non_square(self):
        """Test that non-square NxM matrices get warned when inputted."""
        non_square_matrix = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                [0.82429855, 0.82429855, 0.98175843, 0.98175843],
            ]
        )

        with pytest.warns(Warning, match="The input matrix should be of shape NxN"):
            FABLE(non_square_matrix, tol=0.01)

    def test_padding_for_non_square(self):
        """Test that non-square NxM matrices get padded with zeroes to reach NxN size."""
        non_square_matrix = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                [0.82429855, 0.82429855, 0.98175843, 0.98175843],
            ]
        )

        op = FABLE(non_square_matrix, tol=0.01)
        data = op._flatten()
        assert data[0][0].shape == (4, 4)

    def test_warning_for_not_power(self):
        """Test that matrices with dimensions N that are not a power of 2 get warned."""
        two_by_three_array = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329],
            ]
        )

        with pytest.warns(Warning, match="The input matrix should be of shape NxN"):
            FABLE(two_by_three_array, tol=0.01)

    def test_padding_for_not_power(self):
        """Test that matrices with dimensions N that are not a power of 2 get padded."""
        two_by_three_array = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329],
            ]
        )

        op = FABLE(two_by_three_array, tol=0.01)
        data = op._flatten()
        assert data[0][0].shape == (4, 4)

    @pytest.mark.jax
    def test_jax(self, input_matrix):
        """Test that the Fable operator matrix is correct for jax."""
        import jax.numpy as jnp

        circuit_default = FABLE(input_matrix, 0)
        jax_matrix = jnp.array(input_matrix)
        circuit_jax = FABLE(jax_matrix, 0)

        assert qml.math.allclose(qml.matrix(circuit_default), qml.matrix(circuit_jax))
        assert qml.math.get_interface(qml.matrix(circuit_jax)) == "jax"

    @pytest.mark.jax
    def test_fable_grad_jax(self, input_matrix):
        """Test that BlockEncode is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")

        @qml.qnode(dev, diff_method="backprop")
        def circuit_default(input_matrix):
            FABLE(input_matrix)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            FABLE(jnp.array(input_matrix))
            return qml.expval(qml.PauliZ(wires=0))

        assert np.allclose(
            jax.grad(circuit_default)(input_matrix), jax.grad(circuit_jax)(input_matrix)
        )
