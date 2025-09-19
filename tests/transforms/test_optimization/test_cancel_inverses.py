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
Unit tests for the optimization transform ``cancel_inverses``.
"""

import pytest
from utils import compare_operation_lists

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.optimization import cancel_inverses
from pennylane.wires import Wires


class TestCancelInverses:
    """Test that adjacent inverse gates are cancelled."""

    def test_one_qubit_cancel_adjacent_self_inverse(self):
        """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qml.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_cancel_followed_adjoint(self):
        """Test that a single-qubit circuit with adjacent adjoint gate cancels."""

        def qfunc():
            qml.S(wires=0)
            qml.adjoint(qml.S)(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qml.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_cancel_preceded_adjoint(self):
        """Test that a single-qubit circuit with adjacent adjoint gate cancels."""

        def qfunc():
            qml.adjoint(qml.S)(wires=0)
            qml.S(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qml.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_no_inverse(self):
        """Test that a one-qubit circuit with a gate in the way does not cancel the inverses."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.RZ(0.3, wires=0)
            qml.Hadamard(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "RZ", "Hadamard"]
        wires_expected = [Wires(0)] * 3
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_no_inverse(self):
        """Test that a two-qubit circuit self-inverse on each qubit does not cancel."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard"] * 2
        wires_expected = [Wires(0), Wires(1)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_inverse_after_cnot(self):
        """Test that a three-qubit circuit with a CNOT still allows cancellation."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.CNOT(wires=[0, 2])
            qml.RZ(0.5, wires=2)
            qml.PauliX(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "CNOT", "RZ"]
        wires_expected = [Wires(0), Wires([0, 2]), Wires(2)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_blocking_cnot(self):
        """Test that a three-qubit circuit with a blocking CNOT causes no cancellation."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.5, wires=2)
            qml.PauliX(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "PauliX", "CNOT", "RZ", "PauliX"]
        wires_expected = [Wires(0), Wires(1), Wires([0, 1]), Wires(2), Wires(1)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_diff_ops_adj_no_cancelled(self):
        """Test that different operations do not cancel."""

        def qfunc():
            qml.RX(0.1, wires=0)
            qml.adjoint(qml.RX)(0.2, wires=0)

        transformed_qfunc = cancel_inverses(qfunc)
        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["RX", "Adjoint(RX)"]
        wires_expected = [Wires(0), Wires(0)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_toffolis(self):
        """Test that Toffolis on different permutations of wires cancel correctly."""

        def qfunc():
            # These two will cancel
            qml.Toffoli(wires=["a", "b", "c"])
            qml.Toffoli(wires=["b", "a", "c"])
            # These three will not cancel
            qml.Toffoli(wires=["a", "b", "c"])
            qml.Toffoli(wires=["a", "c", "b"])
            qml.Toffoli(wires=["a", "c", "d"])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Toffoli"] * 3
        wires_expected = [Wires(["a", "b", "c"]), Wires(["a", "c", "b"]), Wires(["a", "c", "d"])]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_cnot_same_direction(self):
        """Test that two adjacent CNOTs cancel."""

        def qfunc():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 1])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 0

    def test_two_qubits_cnot_opposite_direction(self):
        """Test that two adjacent CNOTs with the control/target flipped do NOT cancel."""

        def qfunc():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["CNOT"] * 2
        wires_expected = [Wires([0, 1]), Wires([1, 0])]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_cz_opposite_direction(self):
        """Test that two adjacent CZ with the control/target flipped do cancel due to symmetry."""

        def qfunc():
            qml.CZ(wires=[0, 1])
            qml.CZ(wires=[1, 0])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qml.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 0


# Example QNode and device for interface testing
dev = qml.device("default.qubit", wires=3)


def qfunc_all_ops(theta):
    """Qfunc with ops."""
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)
    qml.S(wires=1)
    qml.adjoint(qml.S)(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta[0], wires=2)
    qml.PauliX(wires=1)
    qml.CZ(wires=[1, 0])
    qml.RY(theta[1], wires=2)
    qml.CZ(wires=[0, 1])
    return qml.expval(qml.PauliX(0) @ qml.PauliX(2))


transformed_qfunc_all_ops = cancel_inverses(qfunc_all_ops)

expected_op_list = ["PauliX", "CNOT", "RZ", "PauliX", "RY"]
expected_wires_list = [Wires(1), Wires([0, 1]), Wires(2), Wires(1), Wires(2)]


class TestCancelInversesInterfaces:
    """Test that adjacent inverse gates are cancelled in all interfaces."""

    @pytest.mark.autograd
    def test_cancel_inverses_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        input = np.array([0.1, 0.2], requires_grad=True)

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
    def test_cancel_inverses_torch(self):
        """Test QNode and gradient in torch interface."""
        import torch

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        original_input = torch.tensor([0.1, 0.2], requires_grad=True)
        transformed_input = torch.tensor([0.1, 0.2], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qml.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qml.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        tape = qml.workflow.construct_tape(transformed_qnode)(transformed_input)
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    def test_cancel_inverses_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        import tensorflow as tf

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        original_input = tf.Variable([0.1, 0.2])
        transformed_input = tf.Variable([0.1, 0.2])

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
    def test_cancel_inverses_jax(self):
        """Test QNode and gradient in JAX interface."""
        import jax
        from jax import numpy as jnp

        original_qnode = qml.QNode(qfunc_all_ops, dev)
        transformed_qnode = qml.QNode(transformed_qfunc_all_ops, dev)

        input = jnp.array([0.1, 0.2], dtype=jnp.float64)

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
    def test_cancel_inverses_abstract_params(self):
        """Test that the transform does not fail with abstract parameters."""
        import jax

        @jax.jit
        @cancel_inverses
        @qml.qnode(dev)
        def circuit(x):
            qml.adjoint(qml.RX(x + 0.0, 0))
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        res = circuit(jax.numpy.array(0))
        qml.math.allclose(res, 1.0)

    @pytest.mark.jax
    def test_cancel_inverses_abstract_wires(self):
        """Tests that inverses do not cancel across operators with abstract wires."""

        import jax

        @jax.jit
        def f(w):
            tape = qml.tape.QuantumScript([qml.H(0), qml.CNOT([w, 1]), qml.H(0)])
            [tape], _ = cancel_inverses(tape)
            return len(tape.operations)

        @jax.jit
        def f2(w):
            tape = qml.tape.QuantumScript([qml.X(0), qml.X(0), qml.CNOT([w, 1])])
            [tape], _ = cancel_inverses(tape)
            return len(tape.operations)

        assert f(0) == 3
        assert f2(0) == 1


### Tape
with qml.queuing.AnnotatedQueue() as q:
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)
    qml.S(wires=1)
    qml.adjoint(qml.S)(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(0.1, wires=2)
    qml.PauliX(wires=1)
    qml.CZ(wires=[1, 0])
    qml.RY(0.2, wires=2)
    qml.CZ(wires=[0, 1])
    qml.expval(qml.PauliX(0) @ qml.PauliX(2))

tape_circuit = qml.tape.QuantumTape.from_queue(q)


### QFunc
def qfunc_circuit(theta):
    """Qfunc circuit"""
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)
    qml.S(wires=1)
    qml.adjoint(qml.S)(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta[0], wires=2)
    qml.PauliX(wires=1)
    qml.CZ(wires=[1, 0])
    qml.RY(theta[1], wires=2)
    qml.CZ(wires=[0, 1])


### QNode
dev = qml.devices.DefaultQubit()


@qml.qnode(device=dev)
def qnode_circuit(theta):
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)
    qml.S(wires=1)
    qml.adjoint(qml.S)(wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(theta[0], wires=2)
    qml.PauliX(wires=1)
    qml.CZ(wires=[1, 0])
    qml.RY(theta[1], wires=2)
    qml.CZ(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))


class TestTransformDispatch:
    """Test cancel inverses on tape, qfunc and QNode."""

    def test_tape(self):
        """Test the transform on tape."""
        tapes, _ = cancel_inverses(tape_circuit)
        assert len(tapes) == 1
        tape = tapes[0]
        assert len(tape.operations) == 5

    def test_qfunc(self):
        """Test the transform on a qfunc inside a qnode."""

        @qml.qnode(device=dev)
        def new_circuit(a):
            cancel_inverses(qfunc_circuit)(a)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(2))

        tape = qml.workflow.construct_tape(new_circuit)([0.1, 0.2])
        assert len(tape.operations) == 5

    def test_qnode(self):
        """Test the transform on a qnode directly."""
        transformed_qnode = cancel_inverses(qnode_circuit)
        assert not transformed_qnode.transform_program.is_empty()
        assert len(transformed_qnode.transform_program) == 1
        params = [0.1, 0.2]
        res = transformed_qnode(params)
        expected = qnode_circuit(params)
        assert np.allclose(res, expected)

    @pytest.mark.jax
    def test_qnode_diff_jax(self):
        """Test the transform on a qnode directly."""
        import jax

        a = jax.numpy.array([0.1, 0.2])
        transformed_qnode = cancel_inverses(qnode_circuit)
        res = jax.jacobian(transformed_qnode)(a)
        expected = jax.jacobian(qnode_circuit)(a)
        assert jax.numpy.allclose(res, expected)
