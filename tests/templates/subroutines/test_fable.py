# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the FABLE template.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


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

    @pytest.mark.jax
    def test_standard_validity(self, input_matrix):
        """Check the operation using the assert_valid function."""
        op = qml.FABLE(input_matrix, wires=range(5), tol=0.01)
        qml.ops.functions.assert_valid(op)

    @pytest.mark.parametrize(
        ("input", "wires"),
        [
            (np.random.random((2, 2)), 3),
            (
                np.array(
                    [
                        [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                        [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                        [0.82429855, 0.82429855, 0.98175843, 0.98175843],
                        [0.99675093, 0.99675093, 0.83514837, 0.83514837],
                    ]
                ),
                5,
            ),
            (np.random.random((8, 8)), 7),
        ],
    )
    def test_fable_real_input_matrices(self, input, wires):
        """Test that FABLE produces the right circuit given a square, real-valued matrix"""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FABLE(input_matrix=input, wires=range(wires), tol=0)
            return qml.state()

        dim = len(input)
        expected = dim * qml.matrix(circuit, wire_order=range(wires))().real[:dim, :dim]
        assert np.allclose(input, expected)

    def test_fable_imaginary_error(self, input_matrix):
        """Test if a ValueError is raised when imaginary values are passed in."""
        imaginary_matrix = input_matrix.astype(np.complex128)
        imaginary_matrix[0, 0] += 0.00001j  # Add a small imaginary component

        with pytest.raises(
            ValueError, match="Support for imaginary values has not been implemented."
        ):
            qml.FABLE(imaginary_matrix, wires=range(5), tol=0.01)

    def test_fable_wires_error(self, input_matrix):
        """Test if a ValueError is raised when incorrect number of wires are passed in."""

        with pytest.raises(ValueError, match="Number of wires is incorrect"):
            qml.FABLE(input_matrix, wires=range(4), tol=0.01)

    def test_fable_normalization_error(self, input_matrix):
        """Test if a ValueError is raised when the normalization factor is greater than 1."""
        input_matrix[0, 0] += 10

        with pytest.raises(ValueError, match="The subnormalization factor should be lower than 1."):
            qml.FABLE(input_matrix, wires=range(5), tol=0.01)

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
            qml.FABLE(non_square_matrix, wires=range(5), tol=0.01)

    @pytest.mark.filterwarnings("ignore:The input matrix should be of shape NxN")
    def test_padding_for_non_square(self):
        """Test that non-square NxM matrices get padded with zeroes to reach NxN size."""
        # pylint: disable=protected-access
        non_square_matrix = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                [0.82429855, 0.82429855, 0.98175843, 0.98175843],
            ]
        )

        op = qml.FABLE(non_square_matrix, wires=range(5), tol=0.01)
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
            qml.FABLE(two_by_three_array, wires=range(5), tol=0.01)

    @pytest.mark.filterwarnings("ignore:The input matrix should be of shape NxN")
    def test_padding_for_not_power(self):
        """Test that matrices with dimensions N that are not a power of 2 get padded."""
        # pylint: disable=protected-access
        two_by_three_array = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329],
            ]
        )

        op = qml.FABLE(two_by_three_array, wires=range(5), tol=0.01)
        data = op._flatten()
        assert data[0][0].shape == (4, 4)

    @pytest.mark.jax
    def test_jax(self, input_matrix):
        """Test that the Fable operator matrix is correct for jax."""
        import jax.numpy as jnp

        circuit_default = qml.FABLE(input_matrix, wires=range(5), tol=0)
        jax_matrix = jnp.array(input_matrix)
        circuit_jax = qml.FABLE(jax_matrix, wires=range(5), tol=0)

        assert qml.math.allclose(qml.matrix(circuit_default), qml.matrix(circuit_jax))
        assert qml.math.get_interface(qml.matrix(circuit_jax)) == "jax"

    @pytest.mark.autograd
    def test_autograd(self, input_matrix):
        """Test that the Fable operator matrix is correct for autograd."""
        circuit_default = qml.FABLE(input_matrix, wires=range(5), tol=0)
        grad_matrix = pnp.array(input_matrix)
        circuit_grad = qml.FABLE(grad_matrix, wires=range(5), tol=0)

        assert qml.math.allclose(qml.matrix(circuit_default), qml.matrix(circuit_grad))
        assert qml.math.get_interface(qml.matrix(circuit_grad)) == "autograd"

    @pytest.mark.jax
    def test_fable_grad_jax(self, input_matrix):
        """Test that FABLE is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        delta = 0.001
        input_matrix = np.array(
            [
                [-0.5, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_positive_delta = np.array(
            [
                [-0.5 + delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_negative_delta = np.array(
            [
                [-0.5 - delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )

        input_jax_positive_delta = jnp.array(input_positive_delta)
        input_jax_negative_delta = jnp.array(input_negative_delta)
        input_matrix_jax = jnp.array(input_matrix)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            qml.FABLE(input_matrix, wires=range(5), tol=0)
            return qml.expval(qml.PauliZ(wires=0))

        grad_fn = jax.grad(circuit_jax)
        gradient_numeric = (
            circuit_jax(input_jax_positive_delta) - circuit_jax(input_jax_negative_delta)
        ) / (2 * delta)
        gradient_jax = grad_fn(input_matrix_jax)
        assert np.allclose(gradient_numeric, gradient_jax[0, 0], rtol=0.001)

    @pytest.mark.jax
    def test_fable_jax_jit(self, input_matrix):
        """Test that FABLE is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")

        delta = 0.001
        input_matrix = np.array(
            [
                [-0.5, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_positive_delta = np.array(
            [
                [-0.5 + delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_negative_delta = np.array(
            [
                [-0.5 - delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )

        input_jax_positive_delta = jnp.array(input_positive_delta)
        input_jax_negative_delta = jnp.array(input_negative_delta)
        input_matrix_jax = jnp.array(input_matrix)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            qml.FABLE(input_matrix, wires=range(5), tol=0)
            return qml.expval(qml.PauliZ(wires=0))

        jitted_fn = jax.jit(circuit_jax)

        grad_fn = jax.grad(jitted_fn)
        gradient_numeric = (
            circuit_jax(input_jax_positive_delta) - circuit_jax(input_jax_negative_delta)
        ) / (2 * delta)
        gradient_jax = grad_fn(input_matrix_jax)

        assert qml.math.allclose(gradient_numeric, gradient_jax[0, 0], rtol=0.001)
        assert qml.math.allclose(jitted_fn(input_matrix), circuit_jax(input_matrix))

    @pytest.mark.jax
    def test_fable_grad_jax_jit_error(self, input_matrix):
        """Test that FABLE is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        input_matrix_jax = jnp.array(input_matrix)

        @jax.jit
        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            qml.FABLE(input_matrix, wires=range(5), tol=0.01)
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.raises(
            ValueError, match="JIT is not supported for tolerance values greater than 0."
        ):
            circuit_jax(input_matrix_jax)

    @pytest.mark.autograd
    def test_fable_autograd(self, input_matrix):
        """Test that FABLE is differentiable when using autograd."""
        dev = qml.device("default.qubit")

        delta = 0.001
        input_matrix = np.array(
            [
                [-0.5, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_positive_delta = np.array(
            [
                [-0.5 + delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )
        input_negative_delta = np.array(
            [
                [-0.5 - delta, -0.4, 0.6, 0.7],
                [0.9, 0.9, 0.8, 0.9],
                [0.8, 0.7, 0.9, 0.8],
                [0.9, 0.7, 0.8, 0.3],
            ]
        )

        input_autograd_positive_delta = pnp.array(input_positive_delta)
        input_autograd_negative_delta = pnp.array(input_negative_delta)
        input_autograd = pnp.array(input_matrix)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_autograd(input_matrix):
            qml.FABLE(input_matrix, wires=range(5), tol=0)
            return qml.expval(qml.PauliZ(wires=0))

        grad_fn = qml.grad(circuit_autograd)
        gradient_numeric = (
            circuit_autograd(input_autograd_positive_delta)
            - circuit_autograd(input_autograd_negative_delta)
        ) / (2 * delta)
        gradient_autograd = grad_fn(input_autograd)
        assert np.allclose(gradient_numeric, gradient_autograd[0, 0], rtol=0.001)

    def test_default_lightning_devices(self, input_matrix):
        """Test that FABLE executes with the default.qubit and lightning.qubit simulators."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FABLE(input_matrix, wires=range(5), tol=0.01)
            return qml.state()

        dev2 = qml.device("lightning.qubit", wires=range(5))

        @qml.qnode(dev2)
        def circuit_lightning():
            qml.FABLE(input_matrix, wires=range(5), tol=0.01)
            return qml.state()

        expected = (
            len(input_matrix)
            * qml.matrix(circuit, wire_order=range(5))().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )

        lightning = (
            len(input_matrix)
            * qml.matrix(circuit_lightning, wire_order=range(5))().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )
        assert np.allclose(lightning, expected)

    @pytest.mark.filterwarnings("ignore:The input matrix should be of shape NxN")
    @pytest.mark.parametrize(
        ("input", "wires"),
        [
            (np.random.random((1, 2)), 3),
            (np.random.random((1, 1)), 3),
            (np.random.random((2, 1)), 3),
            (np.random.random((3, 2)), 5),
            (np.random.random((4, 2)), 5),
            (np.random.random((2, 3)), 5),
            (np.random.random((2, 4)), 5),
            (np.random.random((3, 4)), 5),
            (np.random.random((4, 3)), 5),
            (np.random.random((3, 5)), 7),
            (np.random.random((3, 6)), 7),
            (np.random.random((3, 7)), 7),
            (np.random.random((4, 5)), 7),
            (np.random.random((5, 5)), 7),
            (np.random.random((6, 5)), 7),
        ],
    )
    def test_variety_of_matrix_shapes(self, input, wires):
        """Test that FABLE runs without error for a variety of input shapes."""
        dev = qml.device("default.qubit")
        s = int(qml.math.ceil(qml.math.log2(max(input.shape))))
        s = max(s, 1)
        dim = 2**s

        @qml.qnode(dev)
        def circuit():
            qml.FABLE(input_matrix=input, wires=range(wires), tol=0)
            return qml.state()

        circuit_mat = dim * qml.matrix(circuit, wire_order=range(wires))().real[:dim, :dim]
        # Test that the matrix was encoded up to a constant
        submat = circuit_mat[: input.shape[0], : input.shape[1]]
        assert np.allclose(input, submat)

    @pytest.mark.parametrize(
        ("input", "wires", "tol"),
        [
            (np.random.random((1, 2)), 3, 0),
            (np.random.random((1, 1)), 3, 1),
            (np.random.random((2, 1)), 3, 0),
            (np.random.random((3, 2)), 5, 1),
            (np.random.random((2, 3)), 5, 1),
            (np.random.random((3, 4)), 5, 1),
            (np.random.random((3, 5)), 7, 1),
            (np.random.random((3, 7)), 7, 1),
            (np.random.random((5, 5)), 7, 1),
        ],
    )
    def test_decomposition_new(self, input, wires, tol):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.FABLE(input_matrix=input, wires=range(wires), tol=tol)

        for rule in qml.list_decomps(qml.FABLE):
            _test_decomposition_rule(op, rule)

    def test_decomposition_new_fixed_input(self):
        """Check the operation using the assert_valid function."""
        matrix = np.array(
            [[0.8488749045779405, 0.6727547394771869], [0.21985217715701366, 0.9938695727819239]]
        )

        op = qml.FABLE(matrix, wires=range(3), tol=0)

        for rule in qml.list_decomps(qml.FABLE):
            _test_decomposition_rule(op, rule)
