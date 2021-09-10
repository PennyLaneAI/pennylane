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
Tests for the QubitUnitary decomposition transforms.
"""

import pytest

from itertools import product

import pennylane as qml
from pennylane import numpy as np

from pennylane.wires import Wires
from pennylane.transforms import unitary_to_rot

from gate_data import I, Z, S, T, H, X

single_qubit_decomps = [
    # First set of gates are diagonal and converted to RZ
    (I, qml.RZ, [0.0]),
    (Z, qml.RZ, [np.pi]),
    (S, qml.RZ, [np.pi / 2]),
    (T, qml.RZ, [np.pi / 4]),
    (qml.RZ(0.3, wires=0).matrix, qml.RZ, [0.3]),
    (qml.RZ(-0.5, wires=0).matrix, qml.RZ, [-0.5]),
    # Next set of gates are non-diagonal and decomposed as Rots
    (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
    (X, qml.Rot, [0.0, np.pi, np.pi]),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix, qml.Rot, [0.2, 0.5, -0.3]),
    (np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix, qml.Rot, [-1.0, 2.0, -3.0]),
]

# A simple quantum function for testing
def qfunc(U):
    qml.Hadamard(wires="a")
    qml.QubitUnitary(U, wires="a")
    qml.CNOT(wires=["b", "a"])


class TestDecomposeSingleQubitUnitaryTransform:
    """Tests to ensure the transform itself works in all interfaces."""

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot(self, U, expected_gate, expected_params):
        """Test that the transform works in the autograd interface."""
        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose(ops[1].parameters, expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_torch(self, U, expected_gate, expected_params):
        """Test that the transform works in the torch interface."""
        torch = pytest.importorskip("torch")

        U = torch.tensor(U, dtype=torch.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([x.detach() for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_tf(self, U, expected_gate, expected_params):
        """Test that the transform works in the Tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        U = tf.Variable(U, dtype=tf.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([x.numpy() for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_unitary_to_rot_jax(self, U, expected_gate, expected_params):
        """Test that the transform works in the JAX interface."""
        jax = pytest.importorskip("jax")

        U = jax.numpy.array(U, dtype=jax.numpy.complex64)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)(U).operations

        assert len(ops) == 3

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        assert isinstance(ops[1], expected_gate)
        assert ops[1].wires == Wires("a")
        assert qml.math.allclose([jax.numpy.asarray(x) for x in ops[1].parameters], expected_params)

        assert isinstance(ops[2], qml.CNOT)
        assert ops[2].wires == Wires(["b", "a"])


# A simple circuit; we will test QubitUnitary on matrices constructed using trainable
# parameters, and RZ/RX are easy to write the matrices for.
def original_qfunc_for_grad(angles):
    qml.Hadamard(wires="a")
    qml.RZ(angles[0], wires="a")
    qml.RX(angles[1], wires="b")
    qml.CNOT(wires=["b", "a"])
    return qml.expval(qml.PauliX(wires="a"))


dev = qml.device("default.qubit", wires=["a", "b"])

angle_pairs = [[0.3, 0.3], [np.pi, -0.65], [0.0, np.pi / 2], [np.pi / 3, 0.0]]
diff_methods = ["parameter-shift", "backprop"]
angle_diff_pairs = list(product(angle_pairs, diff_methods))


class TestQubitUnitaryDifferentiability:
    """Tests to ensure the transform is fully differentiable in all interfaces."""

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot(self, rot_angles, diff_method):
        """Tests differentiability in autograd interface."""

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            Z_mat = np.array([[np.exp(-1j * z / 2), 0.0], [0.0, np.exp(1j * z / 2)]])

            c = np.cos(x / 2)
            s = np.sin(x / 2) * 1j
            X_mat = np.array([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)

        input = np.array(rot_angles, requires_grad=True)
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        original_grad = qml.grad(original_qnode)(input)
        transformed_grad = qml.grad(transformed_qnode)(input)

        assert qml.math.allclose(original_grad, transformed_grad)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_torch(self, rot_angles, diff_method):
        """Tests differentiability in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            # Had to do this in order to make a torch tensor of torch tensors
            Z_mat = torch.stack(
                [
                    torch.exp(-1j * z / 2),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.exp(1j * z / 2),
                ]
            ).reshape(2, 2)

            # Variables need to be complex
            c = torch.cos(x / 2).type(torch.complex64)
            s = torch.sin(x / 2) * 1j

            X_mat = torch.stack([c, -s, -s, c]).reshape(2, 2)

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="torch", diff_method=diff_method
        )
        original_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="torch", diff_method=diff_method
        )
        transformed_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result)

        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_tf(self, rot_angles, diff_method):
        """Tests differentiability in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        def qfunc_with_qubit_unitary(angles):
            z = tf.cast(angles[0], tf.complex128)
            x = tf.cast(angles[1], tf.complex128)

            c = tf.cos(x / 2)
            s = tf.sin(x / 2) * 1j

            Z_mat = tf.convert_to_tensor([[tf.exp(-1j * z / 2), 0.0], [0.0, tf.exp(1j * z / 2)]])
            X_mat = tf.convert_to_tensor([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="tf", diff_method=diff_method
        )
        original_input = tf.Variable(rot_angles, dtype=tf.float64)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="tf", diff_method=diff_method
        )
        transformed_input = tf.Variable(rot_angles, dtype=tf.float64)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result)

        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)

        transformed_grad = tape.gradient(loss, transformed_input)

        # For 64bit values, need to slightly increase the tolerance threshold
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_jax(self, rot_angles, diff_method):
        """Tests differentiability in jax interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        def qfunc_with_qubit_unitary(angles):
            z = angles[0]
            x = angles[1]

            Z_mat = jnp.array([[jnp.exp(-1j * z / 2), 0.0], [0.0, jnp.exp(1j * z / 2)]])

            c = jnp.cos(x / 2)
            s = jnp.sin(x / 2) * 1j
            X_mat = jnp.array([[c, -s], [-s, c]])

            qml.Hadamard(wires="a")
            qml.QubitUnitary(Z_mat, wires="a")
            qml.QubitUnitary(X_mat, wires="b")
            qml.CNOT(wires=["b", "a"])
            return qml.expval(qml.PauliX(wires="a"))

        # Setting the dtype to complex64 causes the gradients to be complex...
        input = jnp.array(rot_angles, dtype=jnp.float64)

        original_qnode = qml.QNode(
            original_qfunc_for_grad, dev, interface="jax", diff_method=diff_method
        )
        original_result = original_qnode(input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="jax", diff_method=diff_method
        )
        transformed_result = transformed_qnode(input)
        assert qml.math.allclose(original_result, transformed_result)

        original_grad = jax.grad(original_qnode)(input)
        transformed_grad = jax.grad(transformed_qnode)(input)
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)
