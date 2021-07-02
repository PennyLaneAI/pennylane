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
from pennylane.transforms.optimization import single_qubit_fusion
from gate_data import I

from utils import (
    _compute_matrix_from_ops_one_qubit,
    _compute_matrix_from_ops_two_qubit,
    _check_matrix_equivalence,
    _compare_operation_lists,
)


class TestSingleQubitFusion:
    """Test that sequences of any single-qubit rotations are fully fused."""

    def test_single_qubit_full_fusion(self):
        """Test that a sequence of single-qubit gates all fuse."""

        def qfunc():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)

        transformed_qfunc = single_qubit_fusion(qfunc)

        original_ops = qml.transforms.make_tape(qfunc)().operations
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(transformed_ops) == 1

        # Compare matrices
        matrix_expected = _compute_matrix_from_ops_one_qubit(original_ops)
        matrix_obtained = _compute_matrix_from_ops_one_qubit(transformed_ops)
        assert _check_matrix_equivalence(matrix_expected, matrix_obtained)

    def test_single_qubit_cancelled_fusion(self):
        """Test if a sequence of single-qubit gates that all cancel yields no operations."""

        def qfunc():
            qml.RZ(0.1, wires=0)
            qml.RX(0.2, wires=0)
            qml.RX(-0.2, wires=0)
            qml.RZ(-0.1, wires=0)

        transformed_qfunc = single_qubit_fusion(qfunc)
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations
        assert len(transformed_ops) == 0

    def test_single_qubit_fusion_multiple_qubits(self):
        """Test if a sequence of single-qubit gates that all cancel yields no operations."""

        def qfunc():
            qml.RZ(0.3, wires="a")
            qml.Rot(0.1, 0.2, 0.3, wires="b")
            qml.RX(0.1, wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.SX(wires="b")
            qml.S(wires="b")

        transformed_qfunc = single_qubit_fusion(qfunc)

        original_ops = qml.transforms.make_tape(qfunc)().operations
        transformed_ops = qml.transforms.make_tape(transformed_qfunc)().operations

        names_expected = ["Rot", "Rot", "CNOT", "Rot"]
        wires_expected = [Wires("a"), Wires("b"), Wires(["b", "a"]), Wires("b")]
        _compare_operation_lists(transformed_ops, names_expected, wires_expected)

        # Check matrix representation
        matrix_expected = _compute_matrix_from_ops_two_qubit(original_ops, ["a", "b"])
        matrix_obtained = _compute_matrix_from_ops_two_qubit(transformed_ops, ["a", "b"])
        assert _check_matrix_equivalence(matrix_expected, matrix_obtained)


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
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc = single_qubit_fusion(qfunc)

expected_op_list = ["Rot", "Rot", "CNOT", "CRY", "CRY", "Rot"]
expected_wires_list = [Wires(0), Wires(1), Wires([1, 2]), Wires([1, 2]), Wires([1, 2]), Wires(1)]


class TestSingleQubitFusionInterfaces:
    """Test that rotation merging works in all interfaces."""

    def test_single_qubit_fusion_autograd(self):
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
        _compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_single_qubit_fusion_torch(self):
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
        _compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_single_qubit_fusion_tf(self):
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
        _compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_single_qubit_fusion_jax(self):
        """Test QNode and gradient in JAX interface."""
        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

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
        _compare_operation_lists(ops, expected_op_list, expected_wires_list)
