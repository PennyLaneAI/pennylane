import copy
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


def test_standard_checks():
    """Run standard validity tests."""
    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelPrep(lcu, control)
    qml.ops.functions.assert_valid(op)

def test_repr():
    """Test the repr method."""
    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelPrep(lcu, control)
    assert repr(op) == "PrepSelPrep(coeffs=(0.25, 0.75), ops=(Z(2), X(1) @ X(2)), control=<Wires = [0]>)"

def manual_circuit(lcu, control):
    coeffs, unitaries = lcu.terms()
    normalized_coeffs = np.sqrt(coeffs) / np.linalg.norm(np.sqrt(coeffs))

    qml.StatePrep(normalized_coeffs, wires=control)
    qml.Select(unitaries, control=control)
    qml.adjoint(qml.StatePrep(normalized_coeffs, wires=control))

    return qml.state()

def prepselprep_circuit(lcu, control):
    qml.PrepSelPrep(lcu, control)
    return qml.state()

def test_decomposition():
    dev = qml.device("default.qubit")
    manual = qml.QNode(manual_circuit, dev)
    prepselprep = qml.QNode(prepselprep_circuit, dev)

    lcu = qml.dot([0.25, 0+0.75j], [qml.Z(2), qml.X(1) @ qml.X(2)])
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 1, 2])(lcu, control=0), 
        qml.matrix(prepselprep, wire_order=[0, 1, 2])(lcu, control=0))

    lcu = qml.dot([1/2, 1/2], [qml.Identity(0), qml.PauliZ(0)])
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 'ancilla'])(lcu, control='ancilla'), 
        qml.matrix(prepselprep, wire_order=[0, 'ancilla'])(lcu, control='ancilla'))

    a = 0.25
    b = 0.75
    A = np.array(
        [[a, 0, 0, b],
         [0, -a, b, 0],
         [0, b, a, 0],
         [b, 0, 0, -a]])
    lcu = qml.pauli_decompose(A)
    coeffs, unitaries = lcu.terms()
    unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in unitaries]
    lcu = qml.dot(coeffs, unitaries)
    assert np.array_equal(
        qml.matrix(manual, wire_order=[0, 1, 2])(lcu, control=0), 
        qml.matrix(prepselprep, wire_order=[0, 1, 2])(lcu, control=0))

def test_copy():
    lcu = qml.dot([1/2, 1/2], [qml.Identity(1), qml.PauliZ(1)])
    op = qml.PrepSelPrep(lcu, control=0)
    op_copy = copy.copy(op)

    assert qml.equal(op, op_copy)

def test_flatten_unflatten():
    lcu = qml.dot([1/2, 1/2], [qml.Identity(1), qml.PauliZ(1)])
    lcu_coeffs, lcu_ops = lcu.terms()

    op = qml.PrepSelPrep(lcu, control=0)
    data, metadata = op._flatten()
    data_coeffs, data_ops = data.terms()

    assert hash(metadata)

    assert len(data) == len(lcu)
    assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(lcu_coeffs, data_coeffs))
    assert all(op1 == op2 for op1, op2 in zip(lcu_ops, data_ops))

    assert metadata == op.control

    new_op = type(op)._unflatten(*op._flatten())
    assert op.lcu == new_op.lcu
    assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(op.coeffs, new_op.coeffs))
    assert all(qml.equal(op1, op2) for op1, op2 in zip(op.ops, new_op.ops))
    assert op.control == new_op.control
    assert op.wires == new_op.wires
    assert op.target_wires == new_op.target_wires
    assert op is not new_op

def test_control_in_ops():
    lcu = qml.dot([1/2, 1/2], [qml.Identity(0), qml.PauliZ(0)])
    with pytest.raises(ValueError, match="Control wires should be different from operation wires."):
        qml.PrepSelPrep(lcu, control=0)

def test_autograd():
    dev = qml.device("default.qubit")

    coeffs = pnp.array([1/2, 1/2], requires_grad=True)
    ops = [qml.Identity(1), qml.PauliZ(1)]

    @qml.qnode(dev)
    def prepselprep(coeffs):
        lcu = qml.ops.LinearCombination(coeffs, ops)
        qml.PrepSelPrep(lcu, control=0)
        return qml.expval(qml.Z(1))

    @qml.qnode(dev)
    def manual(coeffs):
        lcu = qml.ops.LinearCombination(coeffs, ops)
        normalized_coeffs = (qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs)))

        qml.StatePrep(normalized_coeffs, wires=0)
        qml.Select(ops, control=0)
        qml.adjoint(qml.StatePrep(normalized_coeffs, wires=0))
        return qml.expval(qml.Z(1))

    grad_prepselprep = qml.grad(prepselprep)(coeffs)
    grad_manual = qml.grad(manual)(coeffs)

    assert qml.math.allclose(grad_prepselprep, grad_manual)

def test_jax():
    import jax
    import jax.numpy as jnp

    dev = qml.device("default.qubit")
    coeffs = jnp.array([1/2, 1/2])
    ops = [qml.Identity(1), qml.PauliZ(1)]

    @qml.qnode(dev)
    def prepselprep(coeffs):
        lcu = qml.ops.LinearCombination(coeffs, ops)
        qml.PrepSelPrep(lcu, control=0)
        return qml.expval(qml.Z(1))

    @qml.qnode(dev)
    def manual(coeffs):
        lcu = qml.ops.LinearCombination(coeffs, ops)
        normalized_coeffs = (qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs)))

        qml.StatePrep(normalized_coeffs, wires=0)
        qml.Select(ops, control=0)
        qml.adjoint(qml.StatePrep(normalized_coeffs, wires=0))
        return qml.expval(qml.Z(1))

    grad_prepselprep = jax.grad(prepselprep)(coeffs)
    grad_manual = jax.grad(manual)(coeffs)

    assert qml.math.allclose(grad_prepselprep, grad_manual)
