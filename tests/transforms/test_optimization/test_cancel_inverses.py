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

import pennylane as qp
from pennylane import numpy as np
from pennylane.transforms.optimization import cancel_inverses
from pennylane.wires import Wires


class TestCancelInverses:
    """Test that adjacent inverse gates are cancelled."""

    def test_pass_name_defined(self):
        """Test cancel_inverses defines a pass_name."""
        assert cancel_inverses.pass_name == "cancel-inverses"

    def test_one_qubit_cancel_adjacent_self_inverse(self):
        """Test that a single-qubit circuit with adjacent self-inverse gate cancels."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qp.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_cancel_followed_adjoint(self):
        """Test that a single-qubit circuit with adjacent adjoint gate cancels."""

        def qfunc():
            qp.S(wires=0)
            qp.adjoint(qp.S)(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qp.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_cancel_preceded_adjoint(self):
        """Test that a single-qubit circuit with adjacent adjoint gate cancels."""

        def qfunc():
            qp.adjoint(qp.S)(wires=0)
            qp.S(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        new_tape = qp.tape.make_qscript(transformed_qfunc)()

        assert len(new_tape.operations) == 0

    def test_one_qubit_no_inverse(self):
        """Test that a one-qubit circuit with a gate in the way does not cancel the inverses."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.RZ(0.3, wires=0)
            qp.Hadamard(wires=0)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "RZ", "Hadamard"]
        wires_expected = [Wires(0)] * 3
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_no_inverse(self):
        """Test that a two-qubit circuit self-inverse on each qubit does not cancel."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard"] * 2
        wires_expected = [Wires(0), Wires(1)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_inverse_after_cnot(self):
        """Test that a three-qubit circuit with a CNOT still allows cancellation."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.PauliX(wires=1)
            qp.CNOT(wires=[0, 2])
            qp.RZ(0.5, wires=2)
            qp.PauliX(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "CNOT", "RZ"]
        wires_expected = [Wires(0), Wires([0, 2]), Wires(2)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_blocking_cnot(self):
        """Test that a three-qubit circuit with a blocking CNOT causes no cancellation."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.PauliX(wires=1)
            qp.CNOT(wires=[0, 1])
            qp.RZ(0.5, wires=2)
            qp.PauliX(wires=1)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Hadamard", "PauliX", "CNOT", "RZ", "PauliX"]
        wires_expected = [Wires(0), Wires(1), Wires([0, 1]), Wires(2), Wires(1)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_diff_ops_adj_no_cancelled(self):
        """Test that different operations do not cancel."""

        def qfunc():
            qp.RX(0.1, wires=0)
            qp.adjoint(qp.RX)(0.2, wires=0)

        transformed_qfunc = cancel_inverses(qfunc)
        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["RX", "Adjoint(RX)"]
        wires_expected = [Wires(0), Wires(0)]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_three_qubits_toffolis(self):
        """Test that Toffolis on different permutations of wires cancel correctly."""

        def qfunc():
            # These two will cancel
            qp.Toffoli(wires=["a", "b", "c"])
            qp.Toffoli(wires=["b", "a", "c"])
            # These three will not cancel
            qp.Toffoli(wires=["a", "b", "c"])
            qp.Toffoli(wires=["a", "c", "b"])
            qp.Toffoli(wires=["a", "c", "d"])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["Toffoli"] * 3
        wires_expected = [Wires(["a", "b", "c"]), Wires(["a", "c", "b"]), Wires(["a", "c", "d"])]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_cnot_same_direction(self):
        """Test that two adjacent CNOTs cancel."""

        def qfunc():
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 1])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 0

    def test_two_qubits_cnot_opposite_direction(self):
        """Test that two adjacent CNOTs with the control/target flipped do NOT cancel."""

        def qfunc():
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 0])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = ["CNOT"] * 2
        wires_expected = [Wires([0, 1]), Wires([1, 0])]
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_two_qubits_cz_opposite_direction(self):
        """Test that two adjacent CZ with the control/target flipped do cancel due to symmetry."""

        def qfunc():
            qp.CZ(wires=[0, 1])
            qp.CZ(wires=[1, 0])

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        assert len(ops) == 0

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_no_recursive_cancellation_with_recursive_false(self, wrapped):
        """Test that with `recursive=False`, nested pairs of inverses are not cancelled."""

        def qfunc():
            if wrapped:
                qp.X(0)
            qp.S(0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.adjoint(qp.S(0))
            if wrapped:
                qp.Y(0)

        transformed_qfunc = cancel_inverses(qfunc, recursive=False)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        if wrapped:
            names_expected = ["PauliX", "S", "Adjoint(S)", "PauliY"]
            wires_expected = [Wires(0)] * 4
        else:
            names_expected = ["S", "Adjoint(S)"]
            wires_expected = [Wires(0)] * 2
        compare_operation_lists(ops, names_expected, wires_expected)

    @pytest.mark.parametrize("wrapped", [False, True])
    def test_recursive_cancellation_with_recursive_true(self, wrapped):
        """Test that with `recursive=True`, nested pairs of inverses are cancelled."""

        def qfunc():
            if wrapped:
                qp.X(0)
            qp.S(0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.adjoint(qp.S(0))
            if wrapped:
                qp.Y(0)

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        if wrapped:
            names_expected = ["PauliX", "PauliY"]
            wires_expected = [Wires(0)] * 2
        else:
            names_expected = []
            wires_expected = []
        compare_operation_lists(ops, names_expected, wires_expected)

    def test_deep_recursive_cancellation(self):
        """Test that deeply nested pairs are cancelled for ``recursive=True``."""

        def qfunc():
            xs = np.arange(500)
            for x in xs:
                qp.RX(x, 0)
            for x in xs[::-1]:
                qp.adjoint(qp.RX(x, 0))

        transformed_qfunc = cancel_inverses(qfunc)
        ops = qp.tape.make_qscript(transformed_qfunc)().operations

        names_expected = []
        wires_expected = []
        compare_operation_lists(ops, names_expected, wires_expected)

    @pytest.mark.parametrize("adjoint_first", [True, False])
    def test_symmetric_over_all_wires(self, adjoint_first):
        """Test that adjacent adjoint ops are cancelled due to wire symmetry."""

        def qfunc(x):
            if adjoint_first:
                qp.adjoint(qp.MultiRZ(x, [2, 0, 1]))
                qp.MultiRZ(x, [0, 1, 2])
            else:
                qp.MultiRZ(x, [2, 0, 1])
                qp.adjoint(qp.MultiRZ(x, [0, 1, 2]))

        transformed_qfunc = cancel_inverses(qfunc)

        ops = qp.tape.make_qscript(transformed_qfunc)(1.5).operations

        names_expected = []
        wires_expected = []
        compare_operation_lists(ops, names_expected, wires_expected)


# Example QNode and device for interface testing
dev = qp.device("default.qubit", wires=3)


def qfunc_all_ops(theta):
    """Qfunc with ops."""
    qp.Hadamard(wires=0)
    qp.PauliX(wires=1)
    qp.S(wires=1)
    qp.adjoint(qp.S)(wires=1)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    qp.RZ(theta[0], wires=2)
    qp.PauliX(wires=1)
    qp.CZ(wires=[1, 0])
    qp.RY(theta[1], wires=2)
    qp.CZ(wires=[0, 1])
    return qp.expval(qp.PauliX(0) @ qp.PauliX(2))


transformed_qfunc_all_ops = cancel_inverses(qfunc_all_ops)

expected_op_list = ["PauliX", "CNOT", "RZ", "PauliX", "RY"]
expected_wires_list = [Wires(1), Wires([0, 1]), Wires(2), Wires(1), Wires(2)]


class TestCancelInversesInterfaces:
    """Test that adjacent inverse gates are cancelled in all interfaces."""

    @pytest.mark.autograd
    def test_cancel_inverses_autograd(self):
        """Test QNode and gradient in autograd interface."""

        original_qnode = qp.QNode(qfunc_all_ops, dev)
        transformed_qnode = qp.QNode(transformed_qfunc_all_ops, dev)

        input = np.array([0.1, 0.2], requires_grad=True)

        # Check that the numerical output is the same
        assert qp.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qp.math.allclose(
            qp.grad(original_qnode)(input), qp.grad(transformed_qnode)(input)
        )

        # Check operation list
        tape = qp.workflow.construct_tape(transformed_qnode)(input)
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.torch
    def test_cancel_inverses_torch(self):
        """Test QNode and gradient in torch interface."""
        import torch

        original_qnode = qp.QNode(qfunc_all_ops, dev)
        transformed_qnode = qp.QNode(transformed_qfunc_all_ops, dev)

        original_input = torch.tensor([0.1, 0.2], requires_grad=True)
        transformed_input = torch.tensor([0.1, 0.2], requires_grad=True)

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qp.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        original_result.backward()
        transformed_result.backward()

        assert qp.math.allclose(original_input.grad, transformed_input.grad)

        # Check operation list
        tape = qp.workflow.construct_tape(transformed_qnode)(transformed_input)
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.tf
    def test_cancel_inverses_tf(self):
        """Test QNode and gradient in tensorflow interface."""
        import tensorflow as tf

        original_qnode = qp.QNode(qfunc_all_ops, dev)
        transformed_qnode = qp.QNode(transformed_qfunc_all_ops, dev)

        original_input = tf.Variable([0.1, 0.2])
        transformed_input = tf.Variable([0.1, 0.2])

        original_result = original_qnode(original_input)
        transformed_result = transformed_qnode(transformed_input)

        # Check that the numerical output is the same
        assert qp.math.allclose(original_result, transformed_result)

        # Check that the gradient is the same
        with tf.GradientTape() as tape:
            loss = original_qnode(original_input)
        original_grad = tape.gradient(loss, original_input)

        with tf.GradientTape() as tape:
            loss = transformed_qnode(transformed_input)
        transformed_grad = tape.gradient(loss, transformed_input)

        assert qp.math.allclose(original_grad, transformed_grad)

        # Check operation list
        tape = qp.workflow.construct_tape(transformed_qnode)(transformed_input)
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_cancel_inverses_jax(self):
        """Test QNode and gradient in JAX interface."""
        import jax
        from jax import numpy as jnp

        original_qnode = qp.QNode(qfunc_all_ops, dev)
        transformed_qnode = qp.QNode(transformed_qfunc_all_ops, dev)

        input = jnp.array([0.1, 0.2], dtype=jnp.float64)

        # Check that the numerical output is the same
        assert qp.math.allclose(original_qnode(input), transformed_qnode(input))

        # Check that the gradient is the same
        assert qp.math.allclose(
            jax.grad(original_qnode)(input), jax.grad(transformed_qnode)(input)
        )

        # Check operation list
        tape = qp.workflow.construct_tape(transformed_qnode)(input)
        ops = tape.operations
        compare_operation_lists(ops, expected_op_list, expected_wires_list)

    @pytest.mark.jax
    def test_cancel_inverses_abstract_params(self):
        """Test that the transform does not fail with abstract parameters."""
        import jax

        @jax.jit
        @cancel_inverses
        @qp.qnode(dev)
        def circuit(x):
            qp.adjoint(qp.RX(x + 0.0, 0))
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        res = circuit(jax.numpy.array(0))
        qp.math.allclose(res, 1.0)

    @pytest.mark.jax
    def test_cancel_inverses_abstract_wires(self):
        """Tests that inverses do not cancel across operators with abstract wires."""

        import jax

        @jax.jit
        def f(w):
            tape = qp.tape.QuantumScript([qp.H(0), qp.CNOT([w, 1]), qp.H(0)])
            [tape], _ = cancel_inverses(tape)
            return len(tape.operations)

        @jax.jit
        def f2(w):
            tape = qp.tape.QuantumScript([qp.X(0), qp.X(0), qp.CNOT([w, 1])])
            [tape], _ = cancel_inverses(tape)
            return len(tape.operations)

        assert f(0) == 3
        assert f2(0) == 1


### Tape
with qp.queuing.AnnotatedQueue() as q:
    qp.Hadamard(wires=0)
    qp.PauliX(wires=1)
    qp.S(wires=1)
    qp.adjoint(qp.S)(wires=1)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    qp.RZ(0.1, wires=2)
    qp.PauliX(wires=1)
    qp.CZ(wires=[1, 0])
    qp.RY(0.2, wires=2)
    qp.CZ(wires=[0, 1])
    qp.expval(qp.PauliX(0) @ qp.PauliX(2))

tape_circuit = qp.tape.QuantumTape.from_queue(q)


### QFunc
def qfunc_circuit(theta):
    """Qfunc circuit"""
    qp.Hadamard(wires=0)
    qp.PauliX(wires=1)
    qp.S(wires=1)
    qp.adjoint(qp.S)(wires=1)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    qp.RZ(theta[0], wires=2)
    qp.PauliX(wires=1)
    qp.CZ(wires=[1, 0])
    qp.RY(theta[1], wires=2)
    qp.CZ(wires=[0, 1])


### QNode
dev = qp.devices.DefaultQubit()


@qp.qnode(device=dev)
def qnode_circuit(theta):
    qp.Hadamard(wires=0)
    qp.PauliX(wires=1)
    qp.S(wires=1)
    qp.adjoint(qp.S)(wires=1)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    qp.RZ(theta[0], wires=2)
    qp.PauliX(wires=1)
    qp.CZ(wires=[1, 0])
    qp.RY(theta[1], wires=2)
    qp.CZ(wires=[0, 1])
    return qp.expval(qp.PauliZ(0) @ qp.PauliZ(2))


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

        @qp.qnode(device=dev)
        def new_circuit(a):
            cancel_inverses(qfunc_circuit)(a)
            return qp.expval(qp.PauliX(0) @ qp.PauliX(2))

        tape = qp.workflow.construct_tape(new_circuit)([0.1, 0.2])
        assert len(tape.operations) == 5

    def test_qnode(self):
        """Test the transform on a qnode directly."""
        transformed_qnode = cancel_inverses(qnode_circuit)
        assert transformed_qnode.transform_program
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
