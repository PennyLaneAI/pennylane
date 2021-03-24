from functools import partial
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms.control import ctrl, ControlledOperation
from pennylane.tape.tape import expand_tape


def assert_equal_operations(ops1, ops2):
    """Assert that two list of operations are equivalent"""
    assert len(ops1) == len(ops2)
    for op1, op2 in zip(ops1, ops2):
        assert type(op1) == type(op2)
        assert op1.wires == op2.wires
        np.testing.assert_allclose(op1.parameters, op2.parameters)


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
        #Execute controlled version.
        cmake_ops()

    expected = [
        qml.CRX(0.123, wires=[1, 0]),
        qml.CRY(0.456, wires=[1, 2]),
        qml.CRX(0.789, wires=[1, 0]),
        qml.CRot(0.111, 0.222, 0.333, wires=[1, 2]),
        qml.CNOT(wires=[1, 2]),
        qml.CY(wires=[1, 4]),
        qml.CZ(wires=[1, 0]),
    ]
    assert len(tape.operations) == 1
    ctrl_op = tape.operations[0]
    assert isinstance(ctrl_op, ControlledOperation)
    expanded = ctrl_op.expand()
    assert_equal_operations(expanded.operations, expected)


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
    for tape in [tape1, tape2]:
        assert len(tape.operations) == 1
        ctrl_op = tape.operations[0]
        assert isinstance(ctrl_op, ControlledOperation)
        expanded = ctrl_op.expand()
        assert_equal_operations(expanded.operations, expected)

def test_nested_control():
    """Test nested use of control"""
    with QuantumTape() as tape:
        CCX = ctrl(ctrl(qml.PauliX, 7), 3)
        CCX(wires=0)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, ControlledOperation)
    new_tape = expand_tape(tape, 3)
    assert_equal_operations(new_tape.operations, [qml.Toffoli(wires=[7, 3, 0])])

def test_multi_control():
    """Test control with a list of wires."""
    with QuantumTape() as tape:
        CCX = ctrl(qml.PauliX, control=[3, 7])
        CCX(wires=0)
    assert len(tape.operations) == 1
    op = tape.operations[0]
    assert isinstance(op, ControlledOperation)
    new_tape = expand_tape(tape, 3)
    assert_equal_operations(new_tape.operations, [qml.Toffoli(wires=[7, 3, 0])])

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
        qml.RX(np.pi/4.0, wires=2)
        ansatz(params)
        return qml.state()

    params = [0.123, 0.456, 0.789, 1.345]
    circuit1 = qml.qnode(dev)(partial(circuit, ansatz=ctrl(my_ansatz, 2)))
    circuit2 = qml.qnode(dev)(partial(circuit, ansatz=controlled_ansatz))
    res1 = circuit1(params=params)
    res2 = circuit2(params=params)
    np.testing.assert_allclose(res1, res2) 


def test_ctrl_within_ctrl():
    """Test using ctrl on a method that uses ctrl."""
    def ansatz(params):
        qml.RX(params[0], wires=0)
        ctrl(qml.PauliX, control=0)(wires=1)
        qml.RX(params[1], wires=0)

    controlled_ansatz = ctrl(ansatz, 2)

    with QuantumTape() as tape:
        controlled_ansatz([0.123, 0.456])

    tape = expand_tape(tape, 2, stop_at=lambda op: not isinstance(op, ControlledOperation))

    expected = [
        qml.CRX(0.123, wires=[2, 0]),
        qml.Toffoli(wires=[0, 2, 1]),
        qml.CRX(0.456, wires=[2, 0])
    ]
    assert_equal_operations(tape.operations, expected)

def test_diagonal_ctrl():
    """Test ctrl on diagonal gates."""
    with QuantumTape() as tape:
        ctrl(qml.DiagonalQubitUnitary, 1)(np.array([-1.0, 1.0j]), wires=0)
    tape = expand_tape(tape, 3, stop_at=lambda op: not isinstance(op, ControlledOperation))
    assert_equal_operations(
        tape.operations, 
        [
            qml.DiagonalQubitUnitary(
                np.array([1.0, 1.0, -1.0, 1.0j]),
                wires=[1, 0])
        ])

def test_qubit_unitary():
    """Test ctrl on QubitUnitary and ControlledQubitUnitary"""
    with QuantumTape() as tape:
        ctrl(qml.QubitUnitary, 1)(
            np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0), 
            wires=0)

    tape = expand_tape(tape, 3, stop_at=lambda op: not isinstance(op, ControlledOperation))
    assert_equal_operations(
        tape.operations, 
        [
            qml.ControlledQubitUnitary(
                np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                control_wires=1,
                wires=0)
        ])

    with QuantumTape() as tape:
        ctrl(qml.ControlledQubitUnitary, 1)(
            np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
            control_wires=2,
            wires=0)

    tape = expand_tape(tape, 3, stop_at=lambda op: not isinstance(op, ControlledOperation))
    assert_equal_operations(
        tape.operations, 
        [
            qml.ControlledQubitUnitary(
                np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                control_wires=[1, 2],
                wires=0)
        ])


def test_no_control_defined():
    """Test a custom operation with no control transform defined."""
    # QFT has no control rule defined.
    with QuantumTape() as tape:
        ctrl(qml.QFT, 2)(wires=[0, 1])
    tape = expand_tape(tape)
    assert len(tape.operations) == 12
    # Check that all operations are updated to their controlled version.
    for op in tape.operations:
        assert type(op) in {qml.ControlledPhaseShift, qml.Toffoli, qml.CRX, qml.CSWAP}

