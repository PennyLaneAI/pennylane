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

from pennylane.transforms.optimization import commute_through_controls_targets
from utils import (
    compare_operation_lists,
    compute_matrix_from_ops_two_qubit,
    check_matrix_equivalence,
)


class TestCommuteBehindControlsTargets:
    """Test that X rotations are properly pushed behind targets of X-based controlled operations."""

    def test_push_behind_x_gates(self):
        """Test that X-basis gates after controlled-X-type gates on targets get pushed behind."""

        def qfunc():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        transformed_qfunc = commute_through_controls_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CNOT", "Toffoli", "PauliX", "RX", "CRX", "SX", "PauliX"]
        wires_expected = [
            Wires([0, 2]),
            Wires([0, 1, 2]),
            Wires(2),
            Wires(2),
            Wires([0, 1]),
            Wires(1),
            Wires(1),
        ]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_dont_push_x_behind_controls(self):
        """Test that X-basis gates after controlled-X-type gates on controls do not get
        pushed behind.
        """

        def qfunc():
            qml.PauliX(wires="a")
            qml.CNOT(wires=["a", "c"])
            qml.RX(0.2, wires="a")
            qml.Toffoli(wires=["c", "a", "b"])

        transformed_qfunc = commute_through_controls_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["PauliX", "CNOT", "RX", "Toffoli"]
        wires_expected = [Wires("a"), Wires(["a", "c"]), Wires("a"), Wires(["c", "a", "b"])]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_push_behind_y_gates(self):
        """Test that Y-basis gates after controlled-Y-type gates on targets get pushed behind."""

        def qfunc():
            qml.PauliY(wires=2)
            qml.CRY(-0.5, wires=["a", 2])
            qml.CNOT(wires=[1, 2])
            qml.RY(0.3, wires=1)
            qml.CY(wires=["a", 1])

        transformed_qfunc = commute_through_controls_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CRY", "PauliY", "CNOT", "CY", "RY"]
        wires_expected = [Wires(["a", 2]), Wires(2), Wires([1, 2]), Wires(["a", 1]), Wires(1)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_dont_push_y_behind_controls(self):
        """Test that Y-basis gates after controlled-Y-type gates on controls do not get
        pushed behind.
        """

        def qfunc():
            qml.CRY(-0.2, wires=["a", 2])
            qml.PauliY(wires="a")
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=["a", 1])
            qml.RY(0.3, wires="a")

        transformed_qfunc = commute_through_controls_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CRY", "PauliY", "CNOT", "CY", "RY"]
        wires_expected = [Wires(["a", 2]), Wires("a"), Wires([1, 2]), Wires(["a", 1]), Wires("a")]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_push_behind_z_gates(self):
        """Test that Z-basis gates after controlled-Z-type gates on controls *and*
        targets get pushed behind."""

        def qfunc():
            qml.PauliZ(wires=2)
            qml.S(wires=0)
            qml.CZ(wires=[0, 2])

            qml.CNOT(wires=[0, 1])

            qml.PhaseShift(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)
            qml.CRZ(0.5, wires=[0, 1])

        transformed_qfunc = commute_through_controls_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["CZ", "PauliZ", "CNOT", "PhaseShift", "CRZ", "S", "T", "PauliZ"]
        wires_expected = (
            [Wires([0, 2]), Wires(2), Wires([0, 1]), Wires(2)] + [Wires([0, 1])] + [Wires(0)] * 3
        )

        compare_operation_lists(ops, names_expected, wires_expected)

    def test_push_behind_mixed_with_matrix(self):
        """Test that Z-basis gates after controlled-Z-type gates on controls *and*
        targets get pushed behind."""

        def qfunc():
            qml.PauliX(wires=1)
            qml.S(wires=0)
            qml.CZ(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.PauliY(wires=1)
            qml.CRY(0.5, wires=[1, 0])
            qml.PhaseShift(0.2, wires=0)
            qml.PauliY(wires=1)
            qml.T(wires=0)
            qml.RZ(0.2, wires=0)
            qml.PauliX(wires=1)
            qml.CRY(0.2, wires=[1, 0])

        transformed_qfunc = commute_through_controls_targets(qfunc)

        original_ops = qml.transforms.make_tape(qfunc)().operations
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(original_ops) == len(transformed_ops)

        # Compare matrices
        matrix_expected = compute_matrix_from_ops_two_qubit(original_ops, wire_order=[0, 1])
        matrix_obtained = compute_matrix_from_ops_two_qubit(transformed_ops, wire_order=[0, 1])
        assert check_matrix_equivalence(matrix_expected, matrix_obtained)


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)


def qfunc(theta):
    qml.PauliX(wires=2)
    qml.S(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliY(wires=1)
    qml.CRY(theta[0], wires=[2, 1])
    qml.PhaseShift(theta[1], wires=0)
    qml.T(wires=0)
    qml.Toffoli(wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(0))


transformed_qfunc = commute_through_controls_targets(qfunc)

expected_op_list = ["PauliX", "CNOT", "CRY", "PauliY", "Toffoli", "S", "PhaseShift", "T"]
expected_wires_list = [
    Wires(2),
    Wires([0, 1]),
    Wires([2, 1]),
    Wires(1),
    Wires([0, 1, 2]),
    Wires(0),
    Wires(0),
    Wires(0),
]


class TestCommuteThroughControlsTargetsInterfaces:
    """Test that single-qubit gates can be pushed through controlled gates in all interfaces."""

    def test_commute_through_controls_targets_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc, dev)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        input = np.array([0.1, 0.2], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_commute_through_controls_targets_torch(self):
        """Test QNode and gradient in torch interface."""
        torch = pytest.importorskip("torch", minversion="1.8")

        original_qnode = qml.QNode(qfunc, dev, interface="torch")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="torch")

        original_input = torch.tensor([1.2, -0.35], requires_grad=True)
        transformed_input = torch.tensor([1.2, -0.35], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_commute_x_behind_targets_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        tf = pytest.importorskip("tensorflow")

        original_qnode = qml.QNode(qfunc, dev, interface="tf")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="tf")

        original_input = tf.Variable([0.8, -0.6])
        transformed_input = tf.Variable([0.8, -0.6])

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

    def test_commute_x_behind_targets_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        original_qnode = qml.QNode(qfunc, dev, interface="jax")
        transformed_qnode = qml.QNode(transformed_qfunc, dev, interface="jax")

        input = jnp.array([0.3, 0.4], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        ops = transformed_qnode.qtape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)
