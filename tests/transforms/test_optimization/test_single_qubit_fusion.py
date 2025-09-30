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
Unit tests for the optimization transform ``single_qubit_fusion``.
"""
import pytest
from utils import check_matrix_equivalence, compare_operation_lists

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.optimization import single_qubit_fusion
from pennylane.wires import Wires


class TestSingleQubitFusion:
    """Test that sequences of any single-qubit rotations are fully fused."""

    # pylint:disable=too-many-function-args

    def test_single_qubit_full_fusion(self):
        """Test that a sequence of single-qubit gates all fuse."""

        def qfunc():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)

        transformed_qfunc = single_qubit_fusion(qfunc)

        # Compare matrices
        matrix_expected = qml.matrix(qfunc, [0])()
        matrix_obtained = qml.matrix(transformed_qfunc, [0])()
        assert qml.math.allclose(matrix_obtained, matrix_expected)

    def test_single_qubit_full_fusion_qnode(self):
        """Test that a sequence of single-qubit gates all fuse."""

        @qml.qnode(device=dev)
        def circuit():
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(0.1, 0.2, 0.3, wires=0)
            qml.RX(0.1, wires=0)
            qml.SX(wires=0)
            qml.T(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Compare matrices
        matrix_expected = qml.matrix(circuit)()
        optimized_qnode = single_qubit_fusion(circuit)
        optimized_qnode()
        matrix_obtained = qml.matrix(optimized_qnode)()
        assert qml.math.allclose(matrix_obtained, matrix_expected)

    def test_single_qubit_fusion_no_gates_after(self):
        """Test that gates with nothing after are applied without modification."""

        def qfunc():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=1)

        transformed_qfunc = single_qubit_fusion(qfunc)
        transformed_ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["RZ", "Hadamard", "GlobalPhase"]
        wires_expected = [Wires(0), Wires(1), Wires([])]
        compare_operation_lists(transformed_ops, names_expected, wires_expected)

    def test_single_qubit_cancelled_fusion(self):
        """Test if a sequence of single-qubit gates that all cancel yields no operations."""

        def qfunc():
            qml.RZ(0.1, wires=0)
            qml.RX(0.2, wires=0)
            qml.RX(-0.2, wires=0)
            qml.RZ(-0.1, wires=0)

        transformed_qfunc = single_qubit_fusion(qfunc, atol=1e-7)
        transformed_ops = qml.tape.make_qscript(transformed_qfunc)().operations
        assert len(transformed_ops) == 0

    def test_single_qubit_fusion_not_implemented(self):
        """Test that fusion is correctly skipped for single-qubit gates where
        the rotation angles are not specified."""

        def qfunc():
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            # No rotation angles specified for PauliRot since it is a gate that
            # in principle acts on an arbitrary number of wires.
            qml.PauliRot(0.2, "X", wires=0)
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)

        transformed_qfunc = single_qubit_fusion(qfunc)
        transformed_ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Rot", "PauliRot", "Rot", "GlobalPhase"]
        wires_expected = [Wires(0)] * 3 + [Wires([])]
        compare_operation_lists(transformed_ops, names_expected, wires_expected)

    def test_single_qubit_fusion_exclude_gates(self):
        """Test that fusion is correctly skipped for gates explicitly on an
        exclusion list."""

        def qfunc():
            # Excluded gate at the start
            qml.RZ(0.1, wires=0)
            qml.Hadamard(wires=0)
            qml.PauliX(wires=0)
            qml.RZ(0.1, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=0)
            # Excluded gate after another gate
            qml.RZ(0.1, wires=0)
            qml.PauliX(wires=1)
            qml.PauliZ(wires=1)
            # Excluded gate after multiple others
            qml.RZ(0.2, wires=1)

        transformed_qfunc = single_qubit_fusion(qfunc, exclude_gates=["RZ"])
        transformed_ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["RZ", "Rot", "RZ", "CNOT", "Hadamard", "RZ", "Rot", "RZ", "GlobalPhase"]
        wires_expected = (
            [Wires(0)] * 2
            + [Wires(1)]
            + [Wires([0, 1])]
            + [Wires(0)] * 2
            + [Wires(1)] * 2
            + [Wires([])]
        )
        compare_operation_lists(transformed_ops, names_expected, wires_expected)

        # Compare matrices
        matrix_expected = qml.matrix(qfunc, [0, 1])()

        matrix_obtained = qml.matrix(transformed_qfunc, [0, 1])()
        assert check_matrix_equivalence(matrix_expected, matrix_obtained)

    def test_single_qubit_fusion_multiple_qubits(self):
        """Test that all sequences of single-qubit gates across multiple qubits fuse properly."""

        def qfunc():
            qml.RZ(0.3, wires="a")
            qml.RY(0.5, wires="a")
            qml.Rot(0.1, 0.2, 0.3, wires="b")
            qml.RX(0.1, wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.SX(wires="b")
            qml.S(wires="b")
            qml.PhaseShift(0.3, wires="b")

        transformed_qfunc = single_qubit_fusion(qfunc)
        transformed_ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Rot", "Rot", "CNOT", "Rot", "GlobalPhase"]
        wires_expected = [Wires("a"), Wires("b"), Wires(["b", "a"]), Wires("b"), Wires([])]
        compare_operation_lists(transformed_ops, names_expected, wires_expected)

        # Check matrix representation
        matrix_expected = qml.matrix(qfunc, ["a", "b"])()

        matrix_obtained = qml.matrix(transformed_qfunc, ["a", "b"])()

        assert check_matrix_equivalence(matrix_expected, matrix_obtained)


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)


# Test each of single-qubit, two-qubit, and Rot gates
def qfunc_all_ops(theta):
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


transformed_qfunc_all_ops = single_qubit_fusion(qfunc_all_ops)

expected_op_list = ["Rot", "Rot", "CNOT", "CRY", "CRY", "Rot", "GlobalPhase"]
expected_wires_list = [
    Wires(0),
    Wires(1),
    Wires([1, 2]),
    Wires([1, 2]),
    Wires([1, 2]),
    Wires(1),
    Wires([]),
]


class TestSingleQubitFusionInterfaces:
    """Test that rotation merging works in all interfaces."""

    @pytest.mark.autograd
    def test_single_qubit_fusion_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        input = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            qml.grad(original_qnode)(input), qml.grad(transformed_qnode)(input)
        )

        # Check operation list
        tape = qml.workflow.construct_tape(transformed_qnode)(input)
        compare_operation_lists(tape.operations, expected_op_list, expected_wires_list)

    @pytest.mark.torch
    def test_single_qubit_fusion_torch(self):
        """Test QNode and gradient in torch interface."""
        import torch

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

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
        tape = qml.workflow.construct_tape(transformed_qnode)(transformed_input)
        compare_operation_lists(tape.operations, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    def test_single_qubit_fusion_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        import tensorflow as tf

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

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
        tape = qml.workflow.construct_tape(transformed_qnode)(transformed_input)
        compare_operation_lists(tape.operations, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_single_qubit_fusion_jax(self):
        """Test QNode and gradient in JAX interface."""
        import jax
        from jax import numpy as jnp

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        input = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qml.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        tape = qml.workflow.construct_tape(transformed_qnode)(input)
        compare_operation_lists(tape.operations, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_single_qubit_fusion_jax_jit(self):
        """Test QNode and gradient in JAX interface with JIT."""
        import jax
        from jax import numpy as jnp

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        jitted_qnode = jax.jit(original_qnode)

        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)
        jitted_transformed_qnode = jax.jit(transformed_qnode)

        input = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64)

        # Check that the numerical output is the same
        original_output = original_qnode(input)
        assert qml.math.allclose(jitted_qnode(input), original_output)
        assert qml.math.allclose(transformed_qnode(input), original_output)
        assert qml.math.allclose(jitted_transformed_qnode(input), original_output)

        # Check that the gradients are the same even after jitting
        original_gradient = jax.grad(original_qnode)(input)
        assert qml.math.allclose(jax.grad(jitted_qnode)(input), original_gradient)
        assert qml.math.allclose(jax.grad(transformed_qnode)(input), original_gradient)
        assert qml.math.allclose(jax.grad(jitted_transformed_qnode)(input), original_gradient)

        # Check operation list
        tape = qml.workflow.construct_tape(transformed_qnode)(input)
        compare_operation_lists(tape.operations, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_single_qubit_fusion_abstract_wires(self):
        """Tests that rotations do not merge across operators with abstract wires."""

        import jax

        @jax.jit
        def f(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.RX(0.5, wires=0),
                    qml.CNOT([w, 1]),
                    qml.RY(0.5, wires=0),
                ]
            )
            [tape], _ = single_qubit_fusion(tape)
            return len(tape.operations)

        @jax.jit
        def f2(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.CNOT([w, 1]),
                    qml.RX(0.5, wires=0),
                    qml.RY(0.5, wires=0),
                ]
            )
            [tape], _ = single_qubit_fusion(tape)
            return len(tape.operations)

        assert f(0) == 3
        assert f2(0) == 2
