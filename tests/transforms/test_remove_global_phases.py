"""
Tests for the remove_global_phases transform.
"""

import pytest

import pennylane as qml
from pennylane.transforms import remove_global_phases


def original_qfunc(phi1, phi2):
    qml.Hadamard(wires=1)
    qml.GlobalPhase(phi1, wires=[0, 1])
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.GlobalPhase(phi2, wires=1)
    qml.CNOT(wires=[2, 0])
    return qml.expval(qml.Z(0) @ qml.X(1))


def expected_qfunc():
    qml.Hadamard(wires=1)
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    return qml.expval(qml.Z(0) @ qml.X(1))


def test_global_phase_removal_error():
    """Test that an error is raised if the transform is applied to a circuit with no measurements."""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)], [qml.state()])

    expected_qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)], [qml.state()])
    with pytest.raises(
        qml.QuantumFunctionError, match="The quantum circuit cannot contain a state measurement"
    ):
        (transformed_qscript,), _ = remove_global_phases(qscript)
        qml.assert_equal(transformed_qscript, expected_qscript)


def test_no_global_phase_gate():
    """Test that when the input ``QuantumScript`` has no ``qml.GlobalPhase`` gate, the returned output is exactly the same"""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)], [qml.expval(qml.Z(0))])

    expected_qscript = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.RX(0, 0)], [qml.expval(qml.Z(0))]
    )
    (transformed_qscript,), _ = remove_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


def test_remove_global_phase_gates():
    """Test that when the input ``QuantumScript`` has ``qml.GlobalPhase`` gates, they are removed with the transform."""
    qscript = qml.tape.QuantumScript(
        [qml.GlobalPhase(1.23, 0), qml.Hadamard(0), qml.GlobalPhase(4.56, 0), qml.RX(0, 0)],
        [qml.expval(qml.Z(0))],
    )

    expected_qscript = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.RX(0, 0)], [qml.expval(qml.Z(0))]
    )
    (transformed_qscript,), _ = remove_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


@pytest.mark.autograd
def test_differentiability_autograd():
    """Test that the output of the ``remove_global_phases`` transform is differentiable with autograd"""
    import pennylane.numpy as pnp

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = remove_global_phases(original_qnode)

    phi1 = pnp.array(0.25)
    phi2 = pnp.array(-0.6)
    grad1, grad2 = qml.jacobian(transformed_qnode)(phi1, phi2)

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)


@pytest.mark.jax
@pytest.mark.parametrize("use_jit", [False, True])
def test_differentiability_jax(use_jit):
    """Test that the output of the ``remove_global_phases`` transform is differentiable with JAX"""
    import jax
    import jax.numpy as jnp

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = remove_global_phases(original_qnode)

    if use_jit:
        transformed_qnode = jax.jit(transformed_qnode)

    phi1 = jnp.array(0.25)
    phi2 = jnp.array(-0.6)
    grad1, grad2 = jax.jacobian(transformed_qnode, argnums=[0, 1])(phi1, phi2)

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)


@pytest.mark.torch
def test_differentiability_torch():
    """Test that the output of the ``remove_global_phases`` transform is differentiable with Torch"""
    import torch
    from torch.autograd.functional import jacobian

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = remove_global_phases(original_qnode)

    phi1 = torch.tensor(0.25)
    phi2 = torch.tensor(-0.6)
    grad1, grad2 = jacobian(transformed_qnode, (phi1, phi2))

    zero = torch.tensor(0.0)
    assert qml.math.isclose(grad1, zero)
    assert qml.math.isclose(grad2, zero)


@pytest.mark.tf
def test_differentiability_tensorflow():
    """Test that the output of the ``remove_global_phases`` transform is differentiable with TensorFlow"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)

    phi1 = tf.Variable(0.25, dtype=tf.float64)
    phi2 = tf.Variable(-0.6, dtype=tf.float64)
    with tf.GradientTape() as tape:
        transformed_qnode = remove_global_phases(original_qnode)(phi1, phi2)
    grad1, grad2 = tape.jacobian(transformed_qnode, (phi1, phi2))

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)
