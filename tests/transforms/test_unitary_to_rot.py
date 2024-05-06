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
from itertools import product

import pytest
from gate_data import H, I, S, T, X, Z
from test_optimization.utils import check_matrix_equivalence

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import unitary_to_rot
from pennylane.wires import Wires

typeof_gates_zyz = (qml.RZ, qml.RY, qml.RZ)
single_qubit_decompositions = [
    (I, typeof_gates_zyz, [0.0, 0.0, 0.0]),
    (Z, typeof_gates_zyz, [np.pi / 2, 0.0, np.pi / 2]),
    (S, typeof_gates_zyz, [np.pi / 4, 0.0, np.pi / 4]),
    (T, typeof_gates_zyz, [np.pi / 8, 0.0, np.pi / 8]),
    (H, typeof_gates_zyz, [np.pi, np.pi / 2, 0.0]),
    (X, typeof_gates_zyz, [np.pi / 2, np.pi, 7 * np.pi / 2]),
    (qml.RZ(0.3, wires=0).matrix(), typeof_gates_zyz, [0.15, 0.0, 0.15]),
    (
        qml.RZ(-0.5, wires=0).matrix(),
        typeof_gates_zyz,
        [4 * np.pi - 0.25, 0.0, 4 * np.pi - 0.25],
    ),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix(), typeof_gates_zyz, [0.2, 0.5, 4 * np.pi - 0.3]),
    (
        np.array(
            [
                [0, -9.831019270939975e-01 + 0.1830590094588862j],
                [9.831019270939975e-01 + 0.1830590094588862j, 0],
            ]
        ),
        typeof_gates_zyz,
        [12.382273469673908, np.pi, 0.18409714468526417],
    ),
    (
        np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix(),
        typeof_gates_zyz,
        [4 * np.pi - 1.0, 2.0, 4 * np.pi - 3.0],
    ),
]


# A simple quantum function for testing
def qfunc(U):
    qml.Hadamard(wires="a")
    qml.QubitUnitary(U, wires="a")
    qml.CNOT(wires=["b", "a"])


class TestDecomposeSingleQubitUnitaryTransform:
    """Tests to ensure the transform itself works in all interfaces."""

    def test_unitary_to_rot_device_transform_programqnode(self):
        """Test for applying the transform on a QNode."""
        dev = qml.device("default.qubit", wires=2)

        U = np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix()

        @qml.qnode(device=dev)
        def circuit(U):
            qml.Hadamard(wires=1)
            qml.QubitUnitary(U, wires=0)
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

        transformed_qnode = unitary_to_rot(circuit)
        res_trans = transformed_qnode(U)
        res = circuit(U)
        assert np.allclose(res_trans, res)

    @pytest.mark.parametrize("U,expected_gates,expected_params", single_qubit_decompositions)
    def test_unitary_to_rot(self, U, expected_gates, expected_params):
        """Test that the transform works in the autograd interface."""
        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)(U).operations

        assert len(ops) == 5

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        for i in range(3):
            assert isinstance(ops[1 + i], expected_gates[i])
            assert ops[1 + i].wires == Wires("a")
            assert qml.math.allclose(
                qml.math.unwrap(ops[1 + i].parameters), expected_params[i], atol=1e-7
            )

        assert isinstance(ops[4], qml.CNOT)
        assert ops[4].wires == Wires(["b", "a"])

    def test_unitary_to_rot_too_big_unitary(self):
        """Test that the transform ignores QubitUnitary instances that are too big
        to decompose."""

        tof = qml.Toffoli(wires=[0, 1, 2]).matrix()

        def qfunc_with_large_qubit_unitary():
            qml.QubitUnitary(H, wires="a")
            qml.QubitUnitary(tof, wires=["a", "b", "c"])

        transformed_qfunc = unitary_to_rot(qfunc_with_large_qubit_unitary)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "RZ"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RY"
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "RZ"
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "QubitUnitary"
        assert ops[3].wires == Wires(["a", "b", "c"])

    @pytest.mark.torch
    @pytest.mark.parametrize("U,expected_gates,expected_params", single_qubit_decompositions)
    def test_unitary_to_rot_torch(self, U, expected_gates, expected_params):
        """Test that the transform works in the torch interface."""
        import torch

        U = torch.tensor(U, dtype=torch.complex128)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)(U).operations

        assert len(ops) == 5

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        for i in range(3):
            assert isinstance(ops[1 + i], expected_gates[i])
            assert ops[1 + i].wires == Wires("a")
            assert qml.math.allclose(
                qml.math.unwrap(ops[1 + i].parameters), expected_params[i], atol=1e-7
            )

        assert isinstance(ops[4], qml.CNOT)
        assert ops[4].wires == Wires(["b", "a"])

    @pytest.mark.tf
    @pytest.mark.parametrize("U,expected_gates,expected_params", single_qubit_decompositions)
    def test_unitary_to_rot_tf(self, U, expected_gates, expected_params):
        """Test that the transform works in the Tensorflow interface."""
        import tensorflow as tf

        U = tf.Variable(U, dtype=tf.complex128)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)(U).operations

        assert len(ops) == 5

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        for i in range(3):
            assert isinstance(ops[1 + i], expected_gates[i])
            assert ops[1 + i].wires == Wires("a")
            assert qml.math.allclose(
                qml.math.unwrap(ops[1 + i].parameters), expected_params[i], atol=1e-7
            )

        assert isinstance(ops[4], qml.CNOT)
        assert ops[4].wires == Wires(["b", "a"])

    @pytest.mark.jax
    @pytest.mark.parametrize("U,expected_gates,expected_params", single_qubit_decompositions)
    def test_unitary_to_rot_jax(self, U, expected_gates, expected_params):
        """Test that the transform works in the JAX interface."""
        import jax

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        transformed_qfunc = unitary_to_rot(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)(U).operations

        assert len(ops) == 5

        assert isinstance(ops[0], qml.Hadamard)
        assert ops[0].wires == Wires("a")

        for i in range(3):
            assert isinstance(ops[1 + i], expected_gates[i])
            assert ops[1 + i].wires == Wires("a")
            assert qml.math.allclose(
                qml.math.unwrap(ops[1 + i].parameters), expected_params[i], atol=1e-7
            )

        assert isinstance(ops[4], qml.CNOT)
        assert ops[4].wires == Wires(["b", "a"])

    @pytest.mark.jax
    @pytest.mark.parametrize("U,expected_gates,expected_params", single_qubit_decompositions)
    def test_unitary_to_rot_jax_jit(self, U, expected_gates, expected_params):
        """Test that the transform works in the JAX interface with JIT."""
        # pylint: disable=unused-argument
        import jax

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        dev = qml.device("default.qubit", wires=["a", "b"])

        def test_qfunc(U):
            qfunc(U)
            return qml.expval(qml.PauliX("a") @ qml.PauliZ("b"))

        original_qnode = qml.QNode(test_qfunc, dev)
        transformed_qnode = qml.QNode(unitary_to_rot(test_qfunc), dev)
        jitted_qnode = jax.jit(transformed_qnode)

        original_result = original_qnode(U)
        transformed_result = transformed_qnode(U)
        jitted_result = jitted_qnode(U)
        assert np.allclose(transformed_result, original_result)
        assert np.allclose(jitted_result, original_result)


# A simple circuit; we will test QubitUnitary on matrices constructed using trainable
# parameters, and RZ/RX are easy to write the matrices for.
def original_qfunc_for_grad(angles):
    qml.Hadamard(wires="a")
    qml.RZ(angles[0], wires="a")
    qml.RX(angles[1], wires="b")
    qml.CNOT(wires=["b", "a"])
    return qml.expval(qml.PauliX(wires="a"))


angle_pairs = [[0.3, 0.3], [np.pi, -0.65], [0.0, np.pi / 2], [np.pi / 3, 0.0]]
diff_methods = ["parameter-shift", "backprop"]
angle_diff_pairs = list(product(angle_pairs, diff_methods))


class TestQubitUnitaryDifferentiability:
    """Tests to ensure the transform is fully differentiable in all interfaces for
    single-qubit unitary decompositions."""

    @pytest.mark.autograd
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

        dev = qml.device("default.qubit", wires=["a", "b"])

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)

        angles = np.array(rot_angles, requires_grad=True)
        assert qml.math.allclose(original_qnode(angles), transformed_qnode(angles), atol=1e-7)

        original_grad = qml.grad(original_qnode)(angles)
        transformed_grad = qml.grad(transformed_qnode)(angles)

        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

    @pytest.mark.torch
    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_torch(self, rot_angles, diff_method):
        """Tests differentiability in torch interface."""
        import torch

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

        dev = qml.device("default.qubit", wires=["a", "b"])

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)
        original_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)
        transformed_input = torch.tensor(rot_angles, dtype=torch.float64, requires_grad=True)
        transformed_result = transformed_qnode(transformed_input)

        assert qml.math.allclose(original_result, transformed_result, atol=1e-7)

        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad, atol=1e-7)

    @pytest.mark.tf
    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_tf(self, rot_angles, diff_method):
        """Tests differentiability in tensorflow interface."""
        import tensorflow as tf

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

        dev = qml.device("default.qubit", wires=["a", "b"])

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)
        original_input = tf.Variable(rot_angles, dtype=tf.float64)
        original_result = original_qnode(original_input)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)
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

    @pytest.mark.jax
    @pytest.mark.parametrize("rot_angles,diff_method", angle_diff_pairs)
    def test_gradient_unitary_to_rot_jax(self, rot_angles, diff_method):
        """Tests differentiability in jax interface."""
        import jax
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
        angles = jnp.array(rot_angles, dtype=jnp.float64)

        dev = qml.device("default.qubit", wires=["a", "b"])

        original_qnode = qml.QNode(original_qfunc_for_grad, dev, diff_method=diff_method)
        original_result = original_qnode(angles)

        transformed_qfunc = unitary_to_rot(qfunc_with_qubit_unitary)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)
        transformed_result = transformed_qnode(angles)
        assert qml.math.allclose(original_result, transformed_result)

        original_grad = jax.grad(original_qnode)(angles)
        transformed_grad = jax.grad(transformed_qnode)(angles)
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

        # Check that we can also JIT
        grad_of_jit = jax.grad(jax.jit(transformed_qnode))(angles)
        jit_of_grad = jax.jit(jax.grad(transformed_qnode))(angles)
        assert np.allclose(original_grad, jit_of_grad, atol=1e-7)
        assert np.allclose(original_grad, grad_of_jit, atol=1e-7)


test_two_qubit_unitaries = [
    # A tensor product of two operations
    [
        [
            0.51742037 - 0.16042302j,
            0.07680202 - 0.28311161j,
            0.66362912 - 0.19848275j,
            0.10182857 - 0.36100112j,
        ],
        [
            -0.29215476 - 0.02638805j,
            0.33026208 - 0.42940231j,
            -0.37321002 - 0.03748534j,
            0.42777752 - 0.5447985j,
        ],
        [
            -0.44919555 + 0.52727826j,
            0.11071838 + 0.35837454j,
            0.35541834 - 0.40882416j,
            -0.0837747 - 0.28112725j,
        ],
        [
            0.33365392 - 0.1713649j,
            -0.06300143 + 0.6898042j,
            -0.26227059 + 0.13139584j,
            0.05467804 - 0.53895241j,
        ],
    ],
    # Random U(4) element
    [
        [
            -0.3391864 - 0.43678034j,
            -0.47208404 + 0.56615881j,
            -0.00288397 + 0.24379941j,
            0.16979202 - 0.25000116j,
        ],
        [
            0.0637868 + 0.3811654j,
            0.1506196 + 0.12326944j,
            -0.59267381 + 0.03717299j,
            0.60727811 - 0.30221148j,
        ],
        [
            -0.6438983 - 0.34218612j,
            0.21084496 - 0.55082361j,
            -0.27556931 - 0.01759285j,
            -0.02951992 - 0.20813943j,
        ],
        [
            -0.06525306 + 0.09415611j,
            -0.04020311 - 0.2631363j,
            0.71028402 - 0.08460563j,
            0.55558715 - 0.30932357j,
        ],
    ],
]


@pytest.mark.parametrize("num_reps", [1, 2, 3, 4, 5])
def test_unitary_to_rot_multiple_two_qubit(num_reps):
    """Test that numerous two-qubit unitaries can be decomposed sequentially."""

    dev = qml.device("default.qubit", wires=2)

    U = np.array(test_two_qubit_unitaries[1], dtype=np.complex128)

    def my_circuit():
        for _ in range(num_reps):
            qml.QubitUnitary(U, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    original_qnode = qml.QNode(my_circuit, dev)
    transformed_qnode = qml.QNode(unitary_to_rot(my_circuit), dev)

    original_matrix = qml.matrix(original_qnode)()
    transformed_matrix = qml.matrix(transformed_qnode)()

    assert check_matrix_equivalence(original_matrix, transformed_matrix, atol=1e-7)


class TestTwoQubitUnitaryDifferentiability:
    """Tests to ensure the transform is fully differentiable in all interfaces
    when the circuit contains two-qubit QubitUnitary operations.

    Note that we are not testing whether we can differentiate w.r.t. the two-qubit
    unitary itself, but rather that the entire pipeline remains differentiable even
    when we are decomposing two-qubit unitaries within.
    """

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_gradient_unitary_to_rot_two_qubit(self, diff_method):
        """Tests differentiability in autograd interface."""
        U0 = np.array(test_two_qubit_unitaries[0], requires_grad=False, dtype=np.complex128)
        U1 = np.array(test_two_qubit_unitaries[1], requires_grad=False, dtype=np.complex128)

        def two_qubit_decomp_qnode(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.QubitUnitary(U0, wires=[0, 1])
            qml.RZ(z, wires=2)
            qml.QubitUnitary(U1, wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        x = np.array(0.1, requires_grad=True)
        y = np.array(0.2, requires_grad=True)
        z = np.array(0.3, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        original_qnode = qml.QNode(two_qubit_decomp_qnode, device=dev, diff_method=diff_method)

        transformed_qfunc = unitary_to_rot(two_qubit_decomp_qnode)
        transformed_qnode = qml.QNode(transformed_qfunc, dev, diff_method=diff_method)

        assert qml.math.allclose(original_qnode(x, y, z), transformed_qnode(x, y, z))

        # 3 normal operations + 18 for the first decomp and 6 for the second
        assert len(transformed_qnode.qtape.operations) == 27

        original_grad = qml.grad(original_qnode)(x, y, z)
        transformed_grad = qml.grad(transformed_qnode)(x, y, z)

        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-6)

    @pytest.mark.torch
    def test_gradient_unitary_to_rot_torch_two_qubit(self):
        """Tests differentiability in torch interface."""
        import torch

        U0 = torch.tensor(test_two_qubit_unitaries[0], requires_grad=False, dtype=torch.complex128)
        U1 = torch.tensor(test_two_qubit_unitaries[1], requires_grad=False, dtype=torch.complex128)

        def two_qubit_decomp_qnode(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.QubitUnitary(U0, wires=[0, 1])
            qml.RZ(z, wires=2)
            qml.QubitUnitary(U1, wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        x = torch.tensor(0.1, requires_grad=True)
        y = torch.tensor(0.2, requires_grad=True)
        z = torch.tensor(0.3, requires_grad=True)

        transformed_x = torch.tensor(0.1, requires_grad=True)
        transformed_y = torch.tensor(0.2, requires_grad=True)
        transformed_z = torch.tensor(0.3, requires_grad=True)

        dev = qml.device("default.qubit", wires=3)

        original_qnode = qml.QNode(
            two_qubit_decomp_qnode, device=dev, interface="torch", diff_method="parameter-shift"
        )

        transformed_qfunc = unitary_to_rot(two_qubit_decomp_qnode)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="torch", diff_method="parameter-shift"
        )

        original_result = original_qnode(x, y, z)

        transformed_result = transformed_qnode(transformed_x, transformed_y, transformed_z)

        assert qml.math.allclose(original_result, transformed_result)

        assert len(transformed_qnode.qtape.operations) == 27

        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(x.grad, transformed_x.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_gradient_unitary_to_rot_tf_two_qubits(self, diff_method):
        """Tests differentiability in tensorflow interface."""
        import tensorflow as tf

        # We have to mark these as constant, otherwise it will try to
        # differentiate with respect to them.
        U0 = tf.constant(test_two_qubit_unitaries[0], dtype=tf.complex128)
        U1 = tf.constant(test_two_qubit_unitaries[1], dtype=tf.complex128)

        def two_qubit_decomp_qnode(x):
            qml.RX(x, wires=0)
            qml.QubitUnitary(U0, wires=[0, 1])
            qml.QubitUnitary(U1, wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        x = tf.Variable(0.1, dtype=tf.float64)

        transformed_x = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=3)

        original_qnode = qml.QNode(
            two_qubit_decomp_qnode, dev, interface="tf", diff_method=diff_method
        )

        original_result = original_qnode(x)

        transformed_qfunc = unitary_to_rot(two_qubit_decomp_qnode)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="tf", diff_method=diff_method
        )

        transformed_result = transformed_qnode(transformed_x)

        assert qml.math.allclose(original_result, transformed_result)

        assert len(transformed_qnode.qtape.operations) == 25

        with tf.GradientTape() as tape:
            loss = original_qnode(x)
        original_grad = tape.gradient(loss, x)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_x)
        transformed_grad = tape.gradient(loss, transformed_x)

        # For 64bit values, need to slightly increase the tolerance threshold
        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-7)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_gradient_unitary_to_rot_two_qubit_jax(self, diff_method):
        """Tests differentiability in jax interface."""
        import jax
        from jax import numpy as jnp

        U0 = jnp.array(test_two_qubit_unitaries[0], dtype=jnp.complex128)
        U1 = jnp.array(test_two_qubit_unitaries[1], dtype=jnp.complex128)

        def two_qubit_decomp_qnode(x):
            qml.RX(x, wires=0)
            qml.QubitUnitary(U0, wires=[0, 1])
            qml.QubitUnitary(U1, wires=[1, 2])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

        x = jnp.array(0.1, dtype=jnp.float64)

        dev = qml.device("default.qubit", wires=3)

        original_qnode = qml.QNode(
            two_qubit_decomp_qnode, device=dev, interface="jax", diff_method=diff_method
        )

        transformed_qfunc = unitary_to_rot(two_qubit_decomp_qnode)
        transformed_qnode = qml.QNode(
            transformed_qfunc, dev, interface="jax", diff_method=diff_method
        )

        assert qml.math.allclose(original_qnode(x), transformed_qnode(x))

        # 1 normal operations + 18 for the first decomp and 6 for the second
        assert len(transformed_qnode.qtape.operations) == 25

        original_grad = jax.grad(original_qnode, argnums=(0))(x)
        transformed_grad = jax.grad(transformed_qnode, argnums=(0))(x)

        assert qml.math.allclose(original_grad, transformed_grad, atol=1e-6)
