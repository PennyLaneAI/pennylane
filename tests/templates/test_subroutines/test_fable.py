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
import pytest
import numpy as np
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane import numpy as pnp


def generate_FABLE_circuit(input_matrix, tol):
    """Circuit that manually creates FABLE gates for tests."""
    alphas = qml.math.arccos(input_matrix).flatten()
    thetas = compute_theta(alphas)

    ancilla = [0]
    s = int(qml.math.log2(qml.math.shape(input_matrix)[0]))
    wires_i = list(range(1, 1 + s))
    wires_j = list(range(1 + s, 1 + 2 * s))

    code = gray_code(2 * qml.math.sqrt(len(input_matrix)))
    n_selections = len(code)

    control_wires = [
        int(qml.math.log2(int(code[i], 2) ^ int(code[(i + 1) % n_selections], 2)))
        for i in range(n_selections)
    ]

    wire_map = dict(enumerate(wires_j + wires_i))

    for w in wires_i:
        qml.Hadamard(w)

    nots = {}
    for theta, control_index in zip(thetas, control_wires):
        if qml.math.is_abstract(theta):
            for c_wire in nots:
                qml.CNOT(wires=[c_wire] + ancilla)
            qml.RY(2 * theta, wires=ancilla)
            nots[wire_map[control_index]] = 1
        else:
            if abs(2 * theta) > tol:
                for c_wire in nots:
                    qml.CNOT(wires=[c_wire] + ancilla)
                qml.RY(2 * theta, wires=ancilla)
                nots = {}
            if wire_map[control_index] in nots:
                del nots[wire_map[control_index]]
            else:
                nots[wire_map[control_index]] = 1

    for c_wire in nots:
        qml.CNOT([c_wire] + ancilla)

    for w_i, w_j in zip(wires_i, wires_j):
        qml.SWAP(wires=[w_i, w_j])

    for w in wires_i:
        qml.Hadamard(w)


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
        op = qml.FABLE(input_matrix, tol=0.01)
        qml.ops.functions.assert_valid(op)

    # pylint: disable=protected-access
    def test_flatten_unflatten(self, input_matrix):
        """Test the flatten and unflatten methods."""
        op = qml.FABLE(input_matrix, tol=0.01)
        data, metadata = op._flatten()
        assert data is op.data
        assert metadata == 0.01
        new_op = type(op)._unflatten(*op._flatten())
        assert qml.equal(op, new_op)

    def test_fable_real(self, input_matrix):
        """Test that FABLE produces the right circuit given a real-valued matrix"""
        ancilla = [0]
        s = int(qml.math.log2(qml.math.shape(input_matrix)[0]))
        wires_i = list(range(1, 1 + s))
        wires_j = list(range(1 + s, 1 + 2 * s))
        wire_order = ancilla + wires_i[::-1] + wires_j[::-1]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FABLE(input_matrix, tol=0.01)
            return qml.state()

        expected = (
            len(input_matrix)
            * qml.matrix(circuit, wire_order=wire_order)().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )
        assert np.allclose(input_matrix, expected)

    def test_fable_imaginary_error(self, input_matrix):
        """Test if a ValueError is raised when imaginary values are passed in."""
        imaginary_matrix = input_matrix.astype(np.complex128)
        imaginary_matrix[0, 0] += 0.00001j  # Add a small imaginary component

        with pytest.raises(
            ValueError, match="Support for imaginary values has not been implemented."
        ):
            qml.FABLE(imaginary_matrix, tol=0.01)

    def test_fable_normalization_error(self, input_matrix):
        """Test if a ValueError is raised when the normalization factor is greater than 1."""
        input_matrix[0, 0] += 10

        with pytest.raises(ValueError, match="The subnormalization factor should be lower than 1."):
            qml.FABLE(input_matrix, tol=0.01)

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
            qml.FABLE(non_square_matrix, tol=0.01)

    @pytest.mark.filterwarnings("ignore:The input matrix should be of shape NxN")
    def test_padding_for_non_square(self):
        """Test that non-square NxM matrices get padded with zeroes to reach NxN size."""
        non_square_matrix = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329, 0.99999329],
                [0.82429855, 0.82429855, 0.98175843, 0.98175843],
            ]
        )

        op = qml.FABLE(non_square_matrix, tol=0.01)
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
            qml.FABLE(two_by_three_array, tol=0.01)

    @pytest.mark.filterwarnings("ignore:The input matrix should be of shape NxN")
    def test_padding_for_not_power(self):
        """Test that matrices with dimensions N that are not a power of 2 get padded."""
        two_by_three_array = np.array(
            [
                [-0.51192128, -0.51192128, 0.6237114],
                [0.97041007, 0.97041007, 0.99999329],
            ]
        )

        op = qml.FABLE(two_by_three_array, tol=0.01)
        data = op._flatten()
        assert data[0][0].shape == (4, 4)

    @pytest.mark.jax
    def test_jax(self, input_matrix):
        """Test that the Fable operator matrix is correct for jax."""
        import jax.numpy as jnp

        circuit_default = qml.FABLE(input_matrix, 0)
        jax_matrix = jnp.array(input_matrix)
        circuit_jax = qml.FABLE(jax_matrix, 0)

        assert qml.math.allclose(qml.matrix(circuit_default), qml.matrix(circuit_jax))
        assert qml.math.get_interface(qml.matrix(circuit_jax)) == "jax"

    @pytest.mark.autograd
    def test_autograd(self, input_matrix):
        """Test that the Fable operator matrix is correct for autograd."""
        circuit_default = qml.FABLE(input_matrix, 0)
        grad_matrix = pnp.array(input_matrix)
        circuit_grad = qml.FABLE(grad_matrix, 0)

        assert qml.math.allclose(qml.matrix(circuit_default), qml.matrix(circuit_grad))
        assert qml.math.get_interface(qml.matrix(circuit_grad)) == "autograd"

    @pytest.mark.jax
    def test_fable_grad_jax(self, input_matrix):
        """Test that FABLE is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")

        input_jax = jnp.array(input_matrix)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_default(input_matrix):
            qml.FABLE(input_matrix, 0)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            qml.FABLE(input_matrix, 0)
            return qml.expval(qml.PauliZ(wires=0))

        grad_fn = jax.grad(circuit_default)
        grads = grad_fn(input_jax)

        grad_fn2 = jax.grad(circuit_jax)
        grads2 = grad_fn2(input_jax)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.jax
    def test_fable_grad_jax_jit(self, input_matrix):
        """Test that FABLE is differentiable when using jax."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")

        input_jax = jnp.array(input_matrix)

        @jax.jit
        @qml.qnode(dev, diff_method="backprop")
        def circuit_default(input_matrix):
            qml.FABLE(input_matrix, 0)
            return qml.expval(qml.PauliZ(wires=0))

        @jax.jit
        @qml.qnode(dev, diff_method="backprop")
        def circuit_jax(input_matrix):
            generate_FABLE_circuit(input_matrix, 0)
            return qml.expval(qml.PauliZ(wires=0))

        grad_fn = jax.grad(circuit_default)
        grads = grad_fn(input_jax)

        grad_fn2 = jax.grad(circuit_jax)
        grads2 = grad_fn2(input_jax)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.autograd
    def test_fable_autograd(self, input_matrix):
        """Test that FABLE is differentiable when using autograd."""
        dev = qml.device("default.qubit")

        input_autograd = pnp.array(input_matrix)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_default(input_matrix):
            qml.FABLE(input_matrix)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev, diff_method="backprop")
        def circuit_autograd(input_matrix):
            generate_FABLE_circuit(input_matrix, 0)
            return qml.expval(qml.PauliZ(wires=0))

        grad_fn = qml.grad(circuit_default)
        grads = grad_fn(input_autograd)

        grad_fn2 = qml.grad(circuit_autograd)
        grads2 = grad_fn2(input_autograd)

        assert qml.math.allclose(grads, grads2)

    def test_default_lightning_devices(self, input_matrix):
        """Test that FABLE executes with the default.qubit and lightning.qubit simulators."""
        ancilla = [0]
        s = int(qml.math.log2(qml.math.shape(input_matrix)[0]))
        wires_i = list(range(1, 1 + s))
        wires_j = list(range(1 + s, 1 + 2 * s))
        wire_order = ancilla + wires_i[::-1] + wires_j[::-1]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.FABLE(input_matrix, tol=0.01)
            return qml.state()

        dev2 = qml.device("lightning.qubit", wires=wire_order)

        @qml.qnode(dev2)
        def circuit_lightning():
            qml.FABLE(input_matrix, tol=0.01)
            return qml.state()

        expected = (
            len(input_matrix)
            * qml.matrix(circuit, wire_order=wire_order)().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )

        lightning = (
            len(input_matrix)
            * qml.matrix(circuit_lightning, wire_order=wire_order)().real[
                0 : len(input_matrix), 0 : len(input_matrix)
            ]
        )
        assert np.allclose(lightning, expected)
