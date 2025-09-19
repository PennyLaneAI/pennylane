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
Unit tests for the optimization transform ``merge_rotations``.
"""
# pylint: disable=too-many-arguments


import pytest
from utils import compare_operation_lists

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.optimization import merge_rotations
from pennylane.wires import Wires


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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == len(expected_ops)

        # Check that all operations and parameter values are as expected
        for op_obtained, op_expected in zip(ops, expected_ops):
            assert op_obtained.name == op_expected.name
            assert np.allclose(op_obtained.parameters, op_expected.parameters)

    def test_rot_gate_cancel(self):
        """Test that two rotation gates get merged to the identity operator (cancel)."""

        def qfunc():
            qml.Rot(-1, 0, 1, wires=0)
            qml.Rot(-1, 0, 1, wires=0)

        transformed_qfunc = merge_rotations(qfunc)
        ops = qml.tape.make_qscript(transformed_qfunc)().operations
        assert not ops

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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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
        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 1
        assert ops[0].name == "RZ"
        assert ops[0].parameters[0] == -1e-7

        # Now try with higher tolerance threshold; the ops should cancel
        transformed_qfunc = merge_rotations(qfunc, atol=1e-5)
        ops = qml.tape.make_qscript(transformed_qfunc)().operations
        assert len(ops) == 0

    @pytest.mark.parametrize(
        ("theta_11", "theta_12", "theta_21", "theta_22", "expected_ops"),
        [
            (0.3, -0.2, 0.5, -0.8, [qml.RX(0.1, wires=0), qml.RY(-0.3, wires=1)]),
            (0.3, -0.3, 0.7, -0.1, [qml.RY(0.6, wires=1)]),
        ],
    )
    def test_two_qubits_merge(self, theta_11, theta_12, theta_21, theta_22, expected_ops):
        """Test that a two-qubit circuit with rotations on different qubits get merged."""

        def qfunc():
            qml.RX(theta_11, wires=0)
            qml.RY(theta_21, wires=1)
            qml.RX(theta_12, wires=0)
            qml.RY(theta_22, wires=1)

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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
            qml.RZ(0.2, wires=[2])

        transformed_qfunc = merge_rotations(qfunc, include_gates=["RX", "CRX"])

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["CRX", "RY", "RY", "RX", "RZ"]
        wires_expected = [Wires([0, 1]), Wires("a"), Wires("a"), Wires(2), Wires(2)]
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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

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

        transformed_qfunc = merge_rotations(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["CRX", "CRX"]
        wires_expected = [Wires(["w1", "w2"]), Wires(["w2", "w1"])]
        compare_operation_lists(ops, names_expected, wires_expected)

        assert ops[0].parameters[0] == 0.2
        assert ops[1].parameters[0] == 0.3

    def test_merge_rotations_non_commuting_observables(self):
        """Test that merge_rotations can be used with non-commuting observables."""

        ops = (qml.RX(0.1, 0), qml.RX(0.3, 0))
        ms = (qml.expval(qml.X(0)), qml.expval(qml.Y(0)))

        tape = qml.tape.QuantumScript(ops, ms, shots=50)
        [out], _ = qml.transforms.merge_rotations(tape)
        expected = qml.tape.QuantumScript((qml.RX(0.4, 0),), ms, shots=50)
        qml.assert_equal(out, expected)


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
    qml.Rot(0.0, 0.0, 0.0, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc_all_ops = merge_rotations(qfunc_all_ops)

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

    @pytest.mark.autograd
    def test_merge_rotations_autograd(self):
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
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.torch
    def test_merge_rotations_torch(self):
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
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    def test_merge_rotations_tf(self):
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
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_merge_rotations_jax(self):
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
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_merge_rotations_jax_jit(self):
        """Test that when using jax.jit, the conditional statement that checks for
        0 rotation angles does not break things."""

        import jax

        @jax.jit
        @qml.qnode(qml.device("default.qubit", wires=["w1", "w2"]), interface="jax")
        @merge_rotations
        def qfunc():
            qml.Rot(*jax.numpy.array([0.1, 0.2, 0.3]), wires=["w1"])
            qml.Rot(*jax.numpy.array([-0.3, -0.2, -0.1]), wires=["w1"])
            qml.CRX(jax.numpy.array(0.2), wires=["w1", "w2"])
            qml.CRX(jax.numpy.array(-0.2), wires=["w1", "w2"])
            return qml.expval(qml.PauliZ("w2"))

        res = qfunc()

        assert qml.math.allclose(res, [1.0])

    @pytest.mark.jax
    def test_merge_rotations_abstract_wires(self):
        """Tests that rotations do not merge across operators with abstract wires."""

        import jax

        @jax.jit
        def f(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.RX(0.5, wires=0),
                    qml.CNOT([w, 1]),
                    qml.RX(0.5, wires=0),
                ]
            )
            [tape], _ = merge_rotations(tape)
            return len(tape.operations)

        @jax.jit
        def f2(w):
            tape = qml.tape.QuantumScript(
                [
                    qml.CNOT([w, 1]),
                    qml.RX(0.5, wires=0),
                    qml.RX(0.5, wires=0),
                ]
            )
            [tape], _ = merge_rotations(tape)
            return len(tape.operations)

        assert f(0) == 3
        assert f2(0) == 2


### Tape
with qml.queuing.AnnotatedQueue() as q:
    qml.Hadamard(wires=0)
    qml.RZ(0.1, wires=0)
    qml.PauliY(wires=1)
    qml.RZ(0.2, wires=0)
    qml.CNOT(wires=[1, 2])
    qml.CRY(0.3, wires=[1, 2])
    qml.PauliZ(wires=0)
    qml.CRY(0.4, wires=[1, 2])
    qml.Rot(0.1, 0.2, 0.3, wires=1)
    qml.Rot(0.2, 0.3, 0.1, wires=1)
    qml.Rot(0.0, 0.0, 0.0, wires=1)
    qml.expval(qml.PauliX(0) @ qml.PauliX(2))

tape_circuit = qml.tape.QuantumTape.from_queue(q)


### QFunc
def qfunc_circuit(theta):
    """Qfunc circuit"""
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


### QNode
dev = qml.devices.DefaultQubit()


@qml.qnode(device=dev)
def qnode_circuit(theta):
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


class TestTransformDispatch:
    """Test cancel inverses on tape, qfunc and QNode."""

    def test_tape(self):
        """Test the transform on tape."""
        tapes, _ = merge_rotations(tape_circuit)
        assert len(tapes) == 1
        ops = tapes[0].operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_qfunc(self):
        """Test the transform on a qfunc inside a qnode."""

        @qml.qnode(device=dev)
        def new_circuit(a):
            merge_rotations(qfunc_circuit)(a)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(2))

        tape = qml.workflow.construct_tape(new_circuit)([0.1, 0.2, 0.3, 0.4])
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    def test_qnode(self):
        """Test the transform on a qnode directly."""
        transformed_qnode = merge_rotations(qnode_circuit)
        assert not transformed_qnode.transform_program.is_empty()
        assert len(transformed_qnode.transform_program) == 1
        res = transformed_qnode([0.1, 0.2, 0.3, 0.4])
        exp_res = qnode_circuit([0.1, 0.2, 0.3, 0.4])
        assert np.allclose(res, exp_res)


@pytest.mark.xfail
def test_merge_rotations_non_commuting_observables():
    """Test that merge_rotations works with non-commuting observables."""

    @qml.transforms.merge_rotations
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RX(-x, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

    res = circuit(0.5)
    assert qml.math.allclose(res[0], 1.0)
    assert qml.math.allclose(res[1], 0.0)
