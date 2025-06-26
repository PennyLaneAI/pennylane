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
Tests for the Select template.
"""
# pylint: disable=protected-access,too-many-arguments,import-outside-toplevel, no-self-use
import copy

import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates.subroutines.select import _unary_select


@pytest.mark.parametrize("num_ops", [3, 10, 15, 16])
def test_standard_checks(num_ops):
    """Run standard validity tests."""
    ops = [qml.PauliX(0) for _ in range(num_ops)]
    control = [1, 2, 3, 4]
    work_wires = [5, 6, 7]

    op = qml.Select(ops, control, work_wires)
    assert op.target_wires == qml.wires.Wires(0)
    qml.ops.functions.assert_valid(op)


def test_repr():
    """Test the repr method."""
    ops = [qml.PauliX(0), qml.PauliY(0)]
    control = [1]

    op = qml.Select(ops, control)
    assert repr(op) == "Select(ops=(X(0), Y(0)), control=Wires([1]))"


class TestSelect:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("ops", "control", "expected_gates", "n_wires"),
        [
            (
                [qml.PauliX(wires=0), qml.PauliY(wires=0)],
                [1],
                [
                    qml.ctrl(qml.PauliX(wires=0), control=1, control_values=0),
                    qml.ctrl(qml.PauliY(wires=0), control=1),
                ],
                2,
            ),
            (
                [qml.PauliX(wires=0), qml.Identity(wires=0), qml.PauliZ(wires=0)],
                [1, 2],
                [
                    qml.ctrl(qml.PauliX(wires=0), control=[1, 2], control_values=[0, 0]),
                    qml.ctrl(qml.PauliZ(wires=0), control=[1, 2], control_values=[1, 0]),
                ],
                3,
            ),
            (
                [
                    qml.PauliX(wires=0),
                    qml.Identity(wires=0),
                    qml.Identity(wires=0),
                    qml.RX(0.3, wires=0),
                ],
                [1, 2],
                [
                    qml.ctrl(qml.PauliX(wires=0), control=[1, 2], control_values=[0, 0]),
                    qml.ctrl(qml.RX(0.3, wires=0), control=[1, 2], control_values=[1, 1]),
                ],
                3,
            ),
            (
                [qml.PauliX(wires="a"), qml.RX(0.7, wires="b")],
                ["c", 1],
                [
                    qml.ctrl(qml.PauliX(wires="a"), control=["c", 1], control_values=[0, 0]),
                    qml.ctrl(qml.RX(0.7, wires="b"), control=["c", 1], control_values=[0, 1]),
                ],
                ["a", "b", "c", 1],
            ),
        ],
    )
    def test_operation_result(self, ops, control, expected_gates, n_wires):
        """Test the correctness of the Select template output."""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit1():
            for wire in control:
                qml.Hadamard(wires=wire)

            qml.Select(ops, control)
            return qml.state()

        @qml.qnode(dev)
        def circuit2():
            for wire in control:
                qml.Hadamard(wires=wire)
            for op in expected_gates:
                qml.apply(op)
            return qml.state()

        assert np.allclose(circuit1(), circuit2())

    @pytest.mark.parametrize(
        ("ops", "control", "expected_gates"),
        [
            (
                [qml.PauliX(wires=0), qml.PauliY(wires=0)],
                [1],
                [
                    qml.ctrl(qml.PauliX(wires=0), control=1, control_values=0),
                    qml.ctrl(qml.PauliY(wires=0), control=1),
                ],
            ),
            (
                [qml.RX(0.5, wires=0), qml.RY(0.7, wires=1)],
                [2],
                [
                    qml.ctrl(qml.RX(0.5, wires=0), control=2, control_values=0),
                    qml.ctrl(qml.RY(0.7, wires=1), control=2),
                ],
            ),
            (
                [
                    qml.RX(0.5, wires=0),
                    qml.RY(0.7, wires=1),
                    qml.RZ(0.3, wires=1),
                    qml.PauliX(wires=2),
                ],
                [3, 4],
                [
                    qml.ctrl(qml.RX(0.5, wires=0), control=[3, 4], control_values=[0, 0]),
                    qml.ctrl(qml.RY(0.7, wires=1), control=[3, 4], control_values=[0, 1]),
                    qml.ctrl(qml.RZ(0.3, wires=1), control=[3, 4], control_values=[1, 0]),
                    qml.ctrl(qml.PauliX(wires=2), control=[3, 4], control_values=[1, 1]),
                ],
            ),
        ],
    )
    def test_queued_ops(self, ops, control, expected_gates):
        """Test the correctness of the Select template queued operations."""
        with qml.tape.OperationRecorder() as recorder:
            qml.Select(ops, control=control)

        select_ops = recorder.expand().operations

        assert [op.name for op in select_ops] == [op.name for op in expected_gates]
        assert [op.wires for op in select_ops] == [op.wires for op in expected_gates]

    @pytest.mark.parametrize(
        ("ops", "control", "expected_gates"),
        [
            (
                [qml.PauliX(wires=0), qml.PauliY(wires=0)],
                [1],
                [
                    qml.ctrl(qml.PauliX(wires=0), control=1, control_values=0),
                    qml.ctrl(qml.PauliY(wires=0), control=1),
                ],
            ),
            (
                [qml.RX(0.5, wires=0), qml.RY(0.7, wires=1)],
                [2],
                [
                    qml.ctrl(qml.RX(0.5, wires=0), control=2, control_values=0),
                    qml.ctrl(qml.RY(0.7, wires=1), control=2),
                ],
            ),
            (
                [
                    qml.RX(0.5, wires=0),
                    qml.RY(0.7, wires=1),
                    qml.RZ(0.3, wires=1),
                    qml.PauliX(wires=2),
                ],
                [3, 4],
                [
                    qml.ctrl(qml.RX(0.5, wires=0), control=[3, 4], control_values=[0, 0]),
                    qml.ctrl(qml.RY(0.7, wires=1), control=[3, 4], control_values=[0, 1]),
                    qml.ctrl(qml.RZ(0.3, wires=1), control=[3, 4], control_values=[1, 0]),
                    qml.ctrl(qml.PauliX(wires=2), control=[3, 4], control_values=[1, 1]),
                ],
            ),
        ],
    )
    def test_decomposition(self, ops, control, expected_gates):
        """Unit test checking that compute_decomposition and decomposition work as expected."""
        op = qml.Select(ops, control=control)
        select_decomposition = op.decomposition()
        select_compute_decomposition = op.compute_decomposition(ops, control)

        for op1, op2 in zip(select_decomposition, expected_gates):
            qml.assert_equal(op1, op2)
        for op1, op2 in zip(select_compute_decomposition, expected_gates):
            qml.assert_equal(op1, op2)

    def test_copy(self):
        """Test that the copy function of Select works correctly."""
        ops = [qml.PauliX(wires=2), qml.RX(0.2, wires=3), qml.PauliY(wires=2), qml.SWAP([2, 3])]
        op = qml.Select(ops, control=[0, 1])
        op_copy = copy.copy(op)

        qml.assert_equal(op, op_copy)


class TestErrorMessages:
    """Test that the correct errors are raised"""

    @pytest.mark.parametrize(
        ("ops", "control", "msg_match"),
        [
            (
                [qml.PauliX(wires=1), qml.PauliY(wires=0), qml.PauliZ(wires=0)],
                [1, 2],
                "Control wires should be different from operation wires.",
            ),
            (
                [qml.PauliX(wires=2)] * 4,
                [1, 2, 3],
                "Control wires should be different from operation wires.",
            ),
            (
                [qml.PauliX(wires="a"), qml.PauliY(wires="b")],
                ["a"],
                "Control wires should be different from operation wires.",
            ),
        ],
    )
    def test_control_in_ops(self, ops, control, msg_match):
        """Test an error is raised when a control wire is in one of the ops"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Select(ops, control)

    @pytest.mark.parametrize(
        ("ops", "control", "msg_match"),
        [
            (
                [qml.PauliX(wires=0), qml.PauliY(wires=0), qml.PauliZ(wires=0)],
                [1],
                r"Not enough control wires \(1\) for the desired number of operations \(3\). At least 2 control wires are required.",
            ),
            (
                [qml.PauliX(wires=0)] * 10,
                [1, 2, 3],
                r"Not enough control wires \(3\) for the desired number of operations \(10\). At least 4 control wires are required.",
            ),
            (
                [qml.PauliX(wires="a"), qml.PauliY(wires="b"), qml.PauliZ(wires="c")],
                [1],
                r"Not enough control wires \(1\) for the desired number of operations \(3\). At least 2 control wires are required.",
            ),
        ],
    )
    def test_too_many_ops(self, ops, control, msg_match):
        """Test that error is raised if more ops are requested than can fit in control wires"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Select(ops, control)


def select_rx_circuit(angles):
    """Circuit that uses Select for tests."""
    qml.Select([qml.RX(angles[0], wires=[1]), qml.RY(angles[1], wires=[1])], control=0)
    return qml.expval(qml.PauliZ(wires=1))


def manual_rx_circuit(angles):
    """Circuit that manually creates Select for tests."""
    qml.ctrl(qml.RX(angles[0], wires=[1]), control=0, control_values=0)
    qml.ctrl(qml.RY(angles[1], wires=[1]), control=0)
    return qml.expval(qml.PauliZ(wires=1))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests the autograd interface."""
        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_select = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_grad = pnp.array(input_default, requires_grad=True)

        grad_fn = qml.grad(circuit_default)
        grads = grad_fn(input_grad)

        grad_fn2 = qml.grad(circuit_select)
        grads2 = grad_fn2(input_grad)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.autograd
    def test_autograd_parameter_shift(self):
        """Tests the autograd interface using the parameter-shift method."""
        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev, diff_method="parameter-shift")
        circuit_select = qml.QNode(select_rx_circuit, dev, diff_method="parameter-shift")

        input_default = [0.5, 0.2]
        input_grad = pnp.array(input_default, requires_grad=True)

        grad_fn = qml.grad(circuit_default)
        grads = grad_fn(input_grad)

        grad_fn2 = qml.grad(circuit_select)
        grads2 = grad_fn2(input_grad)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.tf
    def test_tf(self):
        """Tests the tf interface."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_tf = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_tf = tf.Variable(input_default)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_tf)(input_tf)
        )
        assert qml.math.get_interface(qml.matrix(circuit_tf)(input_tf)) == "tensorflow"

        with tf.GradientTape() as tape:
            res = circuit_default(input_tf)
        grads = tape.gradient(res, [input_tf])

        with tf.GradientTape() as tape2:
            res2 = circuit_tf(input_tf)
        grads2 = tape2.gradient(res2, [input_tf])

        assert qml.math.allclose(grads[0], grads2[0])

    @pytest.mark.torch
    def test_torch(self):
        """Tests the torch interface."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_torch = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_torch = torch.tensor(input_default, requires_grad=True)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_torch)(input_torch)
        )
        assert qml.math.get_interface(qml.matrix(circuit_torch)(input_torch)) == "torch"

        res = circuit_default(input_torch)
        res.backward()
        grads = [input_torch.grad]

        res2 = circuit_torch(input_torch)
        res2.backward()
        grads2 = [input_torch.grad]

        assert qml.math.allclose(grads[0], grads2[0])

    @pytest.mark.jax
    @pytest.mark.slow
    def test_jax(self):
        """Tests the jax interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        input_default = [0.5, 0.2]
        input_jax = jnp.array(input_default)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_jax = qml.QNode(select_rx_circuit, dev)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_jax)(input_jax)
        )
        assert qml.math.get_interface(qml.matrix(circuit_jax)(input_jax)) == "jax"

        grad_fn = jax.grad(circuit_default)
        grads = grad_fn(input_jax)

        grad_fn2 = jax.grad(circuit_jax)
        grads2 = grad_fn2(input_jax)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests jit within the jax interface."""
        import jax

        dev = qml.device("default.qubit", wires=4)
        ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2, 3])]

        @qml.qnode(dev)
        def circuit():
            qml.Select(ops, control=[0, 1])
            return qml.state()

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())


num_controls_and_num_ops = (
    [(nc, i) for nc in range(1, 5) for i in range(1, 2**nc + 1)]
    + [(5, 1), (5, 2), (5, 17), (5, 24), (5, 31)]
    + [(6, 3), (6, 8), (6, 27)]
)


class TestUnaryIterator:
    """Tests for the auxiliary qubit-based unary iterator decomposition of Select."""

    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    def test_no_ops(self, num_controls):
        """Test that the unary iterator does not return any operators for an empty list
        of target operators."""

        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))

        decomp = _unary_select([], control=control, work_wires=work)
        assert decomp == []

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_identity_with_basis_states(self, num_controls, num_ops):
        """Test that the unary iterator is correct by asserting that the identity
        matrix is created by preparing the i-th computational basis state conditioned on the
        i-th basis state in the control qubits."""

        dev = qml.device("default.qubit")

        # Create angle set so that feeding angles[i] into RX on the i-th control wire will
        # yield broadcasted BasisEmbedding (which does not support broadcasting atm)
        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))
        target = list(range(2 * num_controls - 1, 3 * num_controls - 1))

        ops = [qml.BasisEmbedding(i, wires=target) for i in range(num_ops)]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _unary_select(ops, control=control, work_wires=work)
            return qml.probs(target)

        probs = circuit()
        assert np.allclose(probs, np.eye(2**num_controls)[:num_ops])

    @pytest.mark.parametrize(
        ("num_ops", "control", "work", "msg_match"),
        [(9, 4, 1, "Can't use this decomposition")],
    )
    def test_operation_and_test_wires_error(
        self, num_ops, control, work, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        wires = qml.registers({"target": num_ops, "control": control, "work": work})
        ops = [qml.BasisEmbedding(i, wires=wires["target"]) for i in range(num_ops)]

        with pytest.raises(ValueError, match=msg_match):
            _unary_select(ops, control=wires["control"], work_wires=wires["work"])

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_comparison_with_select(self, num_controls, num_ops, seed):
        """Test that the unary iterator is correct by comparing it to the standard Select
        decomposition."""

        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))
        target = [2 * num_controls - 1, 2 * num_controls]

        dev = qml.device("default.qubit", wires=control + work + target)
        unitaries = unitary_group.rvs(4, size=num_ops, random_state=seed)
        if num_ops == 1:
            unitaries = np.array([unitaries])
        ops = [qml.QubitUnitary(U, wires=target) for U in unitaries]
        adj_ops = [qml.QubitUnitary(U.conj().T, wires=target) for U in unitaries]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _unary_select(ops, control=control, work_wires=work)
            qml.Select(adj_ops, control=control, work_wires=None)
            return qml.probs(target)

        probs = circuit()
        exp = np.eye(4)[0]
        assert np.allclose(probs, exp)
