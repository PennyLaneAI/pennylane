from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.tape.tape import expand_tape
from pennylane.ops.op_math import ctrl, Controlled


def test_control_sanity_check():
    """Test that control works on a very standard usecase."""

    def make_ops():
        qml.RX(0.123, wires=0)
        qml.RY(0.456, wires=2)
        qml.RX(0.789, wires=0)
        qml.Rot(0.111, 0.222, 0.333, wires=2),
        qml.PauliX(wires=2)
        qml.PauliY(wires=4)
        qml.PauliZ(wires=0)

    with QuantumTape() as tape:
        cmake_ops = ctrl(make_ops, control=1)
        # Execute controlled version.
        cmake_ops()

    expanded_tape = tape.expand()

    expected = [
        qml.CRX(0.123, wires=[1, 0]),
        qml.CRY(0.456, wires=[1, 2]),
        qml.CRX(0.789, wires=[1, 0]),
        qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]),
        qml.CNOT(wires=[1, 2]),
        qml.CY(wires=[1, 4]),
        qml.CZ(wires=[1, 0]),
    ]
    assert len(tape.operations) == 7
    for op1, op2 in zip(expanded_tape, expected):
        assert qml.equal(op1, op2)


def test_adjoint_of_control():
    """Test adjoint(ctrl(fn)) and ctrl(adjoint(fn))"""

    def my_op(a, b, c):
        qml.RX(a, wires=2)
        qml.RY(b, wires=3)
        qml.RZ(c, wires=0)

    with QuantumTape() as tape1:
        cmy_op_dagger = qml.adjoint(ctrl(my_op, 5))
        # Execute controlled and adjointed version of my_op.
        cmy_op_dagger(0.789, 0.123, c=0.456)

    with QuantumTape() as tape2:
        cmy_op_dagger = ctrl(qml.adjoint(my_op), 5)
        # Execute adjointed and controlled version of my_op.
        cmy_op_dagger(0.789, 0.123, c=0.456)

    expected = [
        qml.CRZ(-0.456, wires=[5, 0]),
        qml.CRY(-0.123, wires=[5, 3]),
        qml.CRX(-0.789, wires=[5, 2]),
    ]
    for tape in [tape1.expand(depth=2), tape2.expand(depth=2)]:
        for op1, op2 in zip(tape, expected):
            assert qml.equal(op1, op2)


def test_nested_control():
    """Test nested use of control"""
    with QuantumTape() as tape:
        CCX = ctrl(ctrl(qml.PauliX, 7), 3)
        CCX(wires=0)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = tape.expand(depth=2)
    assert qml.equal(new_tape[0], qml.Toffoli(wires=[3, 7, 0]))


def test_multi_control():
    """Test control with a list of wires."""
    with QuantumTape() as tape:
        CCX = ctrl(qml.PauliX, control=[3, 7])
        CCX(wires=0)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = tape.expand(depth=1)
    assert qml.equal(new_tape[0], qml.Toffoli(wires=[3, 7, 0]))


def test_control_with_qnode():
    """Test ctrl works when in a qnode cotext."""
    dev = qml.device("default.qubit", wires=3)

    def my_ansatz(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(params[2], wires=1)
        qml.RX(params[3], wires=0)
        qml.CNOT(wires=[1, 0])

    def controlled_ansatz(params):
        qml.CRY(params[0], wires=[2, 0])
        qml.CRY(params[1], wires=[2, 1])
        qml.Toffoli(wires=[2, 0, 1])
        qml.CRX(params[2], wires=[2, 1])
        qml.CRX(params[3], wires=[2, 0])
        qml.Toffoli(wires=[2, 1, 0])

    def circuit(ansatz, params):
        qml.RX(np.pi / 4.0, wires=2)
        ansatz(params)
        return qml.state()

    params = [0.123, 0.456, 0.789, 1.345]
    circuit1 = qml.qnode(dev)(partial(circuit, ansatz=ctrl(my_ansatz, 2)))
    circuit2 = qml.qnode(dev)(partial(circuit, ansatz=controlled_ansatz))
    res1 = circuit1(params=params)
    res2 = circuit2(params=params)
    assert qml.math.allclose(res1, res2)


def test_ctrl_within_ctrl():
    """Test using ctrl on a method that uses ctrl."""

    def ansatz(params):
        qml.RX(params[0], wires=0)
        ctrl(qml.PauliX, control=0)(wires=1)
        qml.RX(params[1], wires=0)

    controlled_ansatz = ctrl(ansatz, 2)

    with QuantumTape() as tape:
        controlled_ansatz([0.123, 0.456])

    tape = tape.expand(2, stop_at=lambda op: not isinstance(op, Controlled))

    expected = [
        qml.CRX(0.123, wires=[2, 0]),
        qml.Toffoli(wires=[2, 0, 1]),
        qml.CRX(0.456, wires=[2, 0]),
    ]
    for op1, op2 in zip(tape, expected):
        assert qml.equal(op1, op2)


def test_diagonal_ctrl():
    """Test ctrl on diagonal gates."""
    with QuantumTape() as tape:
        ctrl(qml.DiagonalQubitUnitary, 1)(np.array([-1.0, 1.0j]), wires=0)
    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))
    assert qml.equal(
        tape[0], qml.DiagonalQubitUnitary(np.array([1.0, 1.0, -1.0, 1.0j]), wires=[1, 0])
    )


def test_qubit_unitary():
    """Test ctrl on QubitUnitary and ControlledQubitUnitary"""
    M = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    with QuantumTape() as tape:
        ctrl(qml.QubitUnitary, 1)(M, wires=0)

    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))

    expected = qml.ControlledQubitUnitary(M, control_wires=1, wires=0)

    assert qml.equal(tape[0], expected)


def test_controlledqubitunitary():
    """Test ctrl on ControlledQubitUnitary."""
    M = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    with QuantumTape() as tape:
        ctrl(qml.ControlledQubitUnitary, 1)(M, control_wires=2, wires=0)

    tape = tape.expand(3, stop_at=lambda op: not isinstance(op, Controlled))

    expected = qml.ControlledQubitUnitary(M, control_wires=[2, 1], wires=0)
    qml.equal(tape[0], expected)


def test_no_control_defined():
    """Test a custom operation with no control transform defined."""
    # QFT has no control rule defined.
    with QuantumTape() as tape:
        ctrl(qml.templates.QFT, 2)(wires=[0, 1])
    tape = tape.expand(depth=3, stop_at=lambda op: not isinstance(op, Controlled))
    assert len(tape.operations) == 12
    # Check that all operations are updated to their controlled version.
    for op in tape.operations:
        assert type(op) in {qml.ControlledPhaseShift, qml.Toffoli, qml.CRX, qml.CSWAP}


def test_no_decomposition_defined():
    """Test that a controlled gate that has no control transform defined,
    as well as no decomposition transformed defined, still works correctly"""

    with QuantumTape() as tape:
        ctrl(qml.CZ, 0)(wires=[1, 2])

    tape = tape.expand()

    assert len(tape.operations) == 1
    assert tape.operations[0].name == "C(CZ)"


def test_controlled_template():
    """Test that a controlled template correctly expands
    on a device that doesn't support it"""

    weights = np.ones([3, 2])

    with QuantumTape() as tape:
        ctrl(qml.templates.BasicEntanglerLayers, 0)(weights, wires=[1, 2])

    tape = expand_tape(tape, depth=2)
    assert len(tape) == 9
    assert all(o.name in {"CRX", "Toffoli"} for o in tape.operations)


def test_controlled_template_and_operations():
    """Test that a combination of controlled templates and operations correctly expands
    on a device that doesn't support it"""

    weights = np.ones([3, 2])

    def ansatz(weights, wires):
        qml.PauliX(wires=wires[0])
        qml.templates.BasicEntanglerLayers(weights, wires=wires)

    with QuantumTape() as tape:
        ctrl(ansatz, 0)(weights, wires=[1, 2])

    tape = tape.expand(depth=2, stop_at=lambda obj: not isinstance(obj, Controlled))
    assert len(tape.operations) == 10
    assert all(o.name in {"CNOT", "CRX", "Toffoli"} for o in tape.operations)


@pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
class TestDifferentiation:
    """Tests for differentiation"""

    @pytest.mark.autograd
    def test_autograd(self, diff_method):
        """Test differentiation using autograd"""
        from pennylane import numpy as pnp

        dev = qml.device("default.qubit", wires=2)
        init_state = pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = pnp.array(0.123, requires_grad=True)
        res = qml.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.torch
    def test_torch(self, diff_method):
        """Test differentiation using torch"""
        import torch

        dev = qml.device("default.qubit", wires=2)
        init_state = torch.tensor([1.0, -1.0], requires_grad=False) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = torch.tensor(0.123, requires_grad=True)
        loss = circuit(b)
        loss.backward()

        res = b.grad.detach()
        expected = np.sin(b.detach() / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["jax", "jax-python", "jax-jit"])
    def test_jax(self, diff_method, jax_interface):
        """Test differentiation using JAX"""

        import jax

        jnp = jax.numpy

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method, interface=jax_interface)
        def circuit(b):
            init_state = np.array([1.0, -1.0]) / np.sqrt(2)
            qml.QubitStateVector(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = jnp.array(0.123)
        res = jax.grad(circuit)(b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)

    @pytest.mark.tf
    def test_tf(self, diff_method):
        """Test differentiation using TF"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        init_state = tf.constant([1.0, -1.0], dtype=tf.complex128) / np.sqrt(2)

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(b):
            qml.QubitStateVector(init_state, wires=0)
            qml.ctrl(qml.RY, control=0)(b, wires=[1])
            return qml.expval(qml.PauliX(0))

        b = tf.Variable(0.123, dtype=tf.float64)

        with tf.GradientTape() as tape:
            loss = circuit(b)

        res = tape.gradient(loss, b)
        expected = np.sin(b / 2) / 2

        assert np.allclose(res, expected)


def test_control_values_sanity_check():
    """Test that control works with control values on a very standard usecase."""

    def make_ops():
        qml.RX(0.123, wires=0)
        qml.RY(0.456, wires=2)
        qml.RX(0.789, wires=0)
        qml.Rot(0.111, 0.222, 0.333, wires=2),
        qml.PauliX(wires=2)
        qml.PauliY(wires=4)
        qml.PauliZ(wires=0)

    with QuantumTape() as tape:
        cmake_ops = ctrl(make_ops, control=1, control_values=0)
        # Execute controlled version.
        cmake_ops()

    expected = [
        qml.PauliX(wires=1),
        qml.CRX(0.123, wires=[1, 0]),
        qml.CRY(0.456, wires=[1, 2]),
        qml.CRX(0.789, wires=[1, 0]),
        qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]),
        qml.CNOT(wires=[1, 2]),
        qml.CY(wires=[1, 4]),
        qml.CZ(wires=[1, 0]),
        qml.PauliX(wires=1),
    ]
    assert len(tape) == 9
    expanded = tape.expand(stop_at=lambda obj: not isinstance(obj, Controlled))
    for op1, op2 in zip(expanded, expected):
        assert qml.equal(op1, op2)


@pytest.mark.parametrize("ctrl_values", [[0, 0], [0, 1], [1, 0], [1, 1]])
def test_multi_control_values(ctrl_values):
    """Test control with a list of wires and control values."""

    def expected_ops(ctrl_val):
        exp_op = []
        ctrl_wires = [3, 7]
        for i, j in enumerate(ctrl_val):
            if not bool(j):
                exp_op.append(qml.PauliX(ctrl_wires[i]))
        exp_op.append(qml.Toffoli(wires=[3, 7, 0]))
        for i, j in enumerate(ctrl_val):
            if not bool(j):
                exp_op.append(qml.PauliX(ctrl_wires[i]))

        return exp_op

    with QuantumTape() as tape:
        CCX = ctrl(qml.PauliX, control=[3, 7], control_values=ctrl_values)
        CCX(wires=0)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, Controlled)
    new_tape = expand_tape(tape, 1)
    for op1, op2 in zip(new_tape, expected_ops(ctrl_values)):
        assert qml.equal(op1, op2)
