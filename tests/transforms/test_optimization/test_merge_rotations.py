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

import pytest
from pennylane import numpy as np

import pennylane as qml
from pennylane.wires import Wires
from pennylane.transforms.optimization import merge_rotations
from utils import compare_operation_lists


class TestMergeRotations:
    """Test that adjacent rotation gates of the same type will add the angles."""

    @pytest.mark.parametrize(
        ("theta_1", "theta_2", "expected_ops"),
        [
            (0.3, -0.2, [qml.RZ(0.1, wires=0)]),
            (0.15, -0.15, []),
        ],
    )
    def test_one_qubit_rotation_merge(self, theta_1, theta_2, expected_ops):
        """Test that a single-qubit circuit with adjacent rotation along the same
        axis either merge, or cancel if the angles sum to 0."""

        def qfunc():
            qml.RZ(theta_1, wires=0)
            qml.RZ(theta_2, wires=0)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        # Check that all operations and parameter values are as expected
        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    @pytest.mark.parametrize(
        ("theta_1", "theta_2", "expected_ops"),
        [
            (0.3, -0.2, [qml.RZ(0.3, wires=0), qml.RZ(-0.2, wires=1)]),
            (0.15, -0.15, [qml.RZ(0.15, wires=0), qml.RZ(-0.15, wires=1)]),
        ],
    )
    def test_two_qubits_rotation_no_merge(self, theta_1, theta_2, expected_ops):
        """Test that a two-qubit circuit with rotations on different qubits
        do not get merged."""

        def qfunc():
            qml.RZ(theta_1, wires=0)
            qml.RZ(theta_2, wires=1)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    def test_two_qubits_rotation_merge_tolerance(self):
        """Test whether tolerance argument is respected for merging."""

        def qfunc():
            qml.RZ(1e-7, wires=0)
            qml.RZ(-2e-7, wires=0)

        # Try with default tolerance; these ops should still be applied
        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 1
        assert ops[0].name == "RZ"
        assert ops[0].parameters[0] == -1e-7

        # Now try with higher tolerance threshold; the ops should cancel
        transformed_qfunc = merge_rotations(atol=1e-5)(qfunc)
        ops = qml.transforms.make_tape(transformed_qfunc)().operations
        assert len(ops) == 0

    @pytest.mark.parametrize(
        ("theta_11", "theta_12", "theta_21", "theta_22", "expected_ops"),
        [
            (0.3, -0.2, 0.5, -0.8, [qml.RX(0.1, wires=0), qml.RY(-0.3, wires=1)]),
            (0.3, -0.3, 0.7, -0.1, [qml.RY(0.6, wires=1)]),
        ],
    )
    def test_two_qubits_rotation_no_merge(
        self, theta_11, theta_12, theta_21, theta_22, expected_ops
    ):
        """Test that a two-qubit circuit with rotations on different qubits get merged."""

        def qfunc():
            qml.RX(theta_11, wires=0)
            qml.RY(theta_21, wires=1)
            qml.RX(theta_12, wires=0)
            qml.RY(theta_22, wires=1)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    @pytest.mark.parametrize(
        ("theta_11", "theta_12", "theta_21", "theta_22", "expected_ops"),
        [
            (0.3, -0.2, 0.5, -0.8, [qml.CRX(0.5, wires=[0, 1]), qml.RY(-1.3, wires=1)]),
            (0.3, 0.3, 0.7, -0.1, [qml.RY(-0.8, wires=1)]),
        ],
    )
    def test_two_qubits_merge_with_adjoint(
        self, theta_11, theta_12, theta_21, theta_22, expected_ops
    ):
        """Test that adjoint rotations on different qubits get merged."""

        def qfunc():
            qml.CRX(theta_11, wires=[0, 1])
            qml.adjoint(qml.RY)(theta_21, wires=2)
            qml.adjoint(qml.CRX)(theta_12, wires=[0, 1])
            qml.RY(theta_22, wires=2)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    def test_two_qubits_merge_gate_subset(self):
        """Test that specifying a subset of operations to include merges correctly."""

        def qfunc():
            qml.CRX(0.1, wires=[0, 1])
            qml.CRX(0.2, wires=[0, 1])
            qml.RY(0.3, wires=["a"])
            qml.RY(0.5, wires=["a"])
            qml.RX(-0.5, wires=[2])
            qml.RX(0.2, wires=[2])

        transformed_qfunc = merge_rotations(include_gates=["RX", "CRX"])(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CRX", "RY", "RY", "RX"]
        wires_expected = [Wires([0, 1]), Wires("a"), Wires("a"), Wires(2)]
        compare_operation_lists(ops, names_expected, wires_expected)

        assert qml.math.isclose(ops[0].parameters[0], 0.3)
        assert qml.math.isclose(ops[1].parameters[0], 0.3)
        assert qml.math.isclose(ops[2].parameters[0], 0.5)
        assert qml.math.isclose(ops[3].parameters[0], -0.3)

    def test_one_qubit_rotation_blocked(self):
        """Test that rotations on one-qubit separated by a "blocking" operation don't merge."""

        def qfunc():
            qml.RX(0.5, wires=0)
            qml.Hadamard(wires=0)
            qml.RX(0.4, wires=0)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["RX", "Hadamard", "RX"]
        wires_expected = [Wires(0)] * 3
        compare_operation_lists(ops, names_expected, wires_expected)

        assert ops[0].parameters[0] == 0.5
        assert ops[2].parameters[0] == 0.4

    def test_two_qubits_rotation_blocked(self):
        """Test that rotations on a two-qubit system separated by a "blocking" operation
        don't merge."""

        def qfunc():
            qml.RX(-0.42, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(0.8, wires=0)

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["RX", "CNOT", "RX"]
        wires_expected = [Wires(0), Wires([0, 1]), Wires(0)]
        compare_operation_lists(ops, names_expected, wires_expected)

        assert ops[0].parameters[0] == -0.42
        assert ops[2].parameters[0] == 0.8

    @pytest.mark.parametrize(
        ("theta_1", "theta_2", "expected_ops"),
        [
            (0.3, -0.2, [qml.CRY(0.1, wires=["w1", "w2"])]),
            (0.15, -0.15, []),
        ],
    )
    def test_controlled_rotation_merge(self, theta_1, theta_2, expected_ops):
        """Test that adjacent controlled rotations on the same wires in same order get merged."""

        def qfunc():
            qml.CRY(theta_1, wires=["w1", "w2"])
            qml.CRY(theta_2, wires=["w1", "w2"])

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        # Check that all operations and parameter values are as expected
        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    def test_controlled_rotation_no_merge(self):
        """Test that adjacent controlled rotations on the same wires in different order don't merge."""

        def qfunc():
            qml.CRX(0.2, wires=["w1", "w2"])
            qml.CRX(0.3, wires=["w2", "w1"])

        transformed_qfunc = merge_rotations()(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CRX", "CRX"]
        wires_expected = [Wires(["w1", "w2"]), Wires(["w2", "w1"])]
        compare_operation_lists(ops, names_expected, wires_expected)

        assert ops[0].parameters[0] == 0.2
        assert ops[1].parameters[0] == 0.3


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)

# Test each of single-qubit, two-qubit, and Rot gates
def qfunc(theta):
    qml.Hadamard(wires=0)
    qml.RZ(theta[0], wires=0)
    qml.PauliY(wires=1)
    qml.RZ(theta[1], wires=0)
    qml.CNOT(wires=[1, 2])
    qml.CRY(theta[2], wires=[1, 2])
    qml.PauliZ(wires=0)
    qml.CRY(theta[3], wires=[1, 2])
    qml.Rot(theta[0], theta[1], theta[2], wires=1)
    qml.Rot(theta[2], theta[3], theta[0], wires=1)
    qml.Rot(0.0, 0.0, 0.0, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc = merge_rotations()(qfunc)

expected_op_list = ["Hadamard", "RZ", "PauliY", "CNOT", "CRY", "PauliZ", "Rot"]
expected_wires_list = [
    Wires(0),
    Wires(0),
    Wires(1),
    Wires([1, 2]),
    Wires([1, 2]),
    Wires(0),
    Wires(1),
]


class TestMergeRotationsInterfaces:
    """Test that rotation merging works in all interfaces."""

    def test_merge_rotations_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        input = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_merge_rotations_torch(self):
        """Test QNode and gradient in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        transformed_input = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result.detach().numpy())

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_merge_rotations_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        original_qnode = qml.QNode(qfunc, dev, interface="tf")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="tf")

        original_input = tf.Variable([0.1, 0.2, 0.3, 0.4])
        transformed_input = tf.Variable([0.1, 0.2, 0.3, 0.4])

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)
        transformed_grad = tape.gradient(loss, transformed_input)

        assert qml.math.allclose(original_grad, transformed_grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_merge_rotations_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        original_qnode = qml.QNode(qfunc, dev, interface="jax")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="jax")

        input = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)
