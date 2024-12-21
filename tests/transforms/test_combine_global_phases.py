"""
Tests for the combine_global_phases transform.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import combine_global_phases


def original_qfunc(phi1, phi2, return_state=False):
    qml.Hadamard(wires=1)
    qml.GlobalPhase(phi1, wires=[0, 1])
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.GlobalPhase(phi2, wires=1)
    qml.CNOT(wires=[2, 0])
    if return_state:
        return qml.state()
    return qml.expval(qml.Z(0) @ qml.X(1))


def expected_qfunc(phi1, phi2, return_state=False):
    qml.Hadamard(wires=1)
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    qml.GlobalPhase(phi1 + phi2)
    if return_state:
        return qml.state()
    return qml.expval(qml.Z(0) @ qml.X(1))


def test_no_global_phase_gate():
    """Test that when the input ``QuantumScript`` has no ``qml.GlobalPhase`` gate, the returned output is exactly the same"""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)])

    expected_qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)])
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


def test_single_global_phase_gate():
    """Test that when the input ``QuantumScript`` has a single ``qml.GlobalPhase`` gate, the returned output has an equivalent
    ``qml.GlobalPhase`` operation appended at the end"""
    phi = 1.23
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.GlobalPhase(phi, 0), qml.RX(0, 0)])

    expected_qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0), qml.GlobalPhase(phi)])
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


def test_multiple_global_phase_gates():
    """Test that when the input ``QuantumScript`` has multiple ``qml.GlobalPhase`` gates, the returned output has an equivalent
    single ``qml.GlobalPhase`` operation appended at the end with a total phase being equal to the sum of each original global phase
    """
    phi1 = 1.23
    phi2 = 4.56
    qscript = qml.tape.QuantumScript(
        [qml.GlobalPhase(phi1, 0), qml.Hadamard(0), qml.GlobalPhase(phi2, 0), qml.RX(0, 0)]
    )

    expected_qscript = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.RX(0, 0), qml.GlobalPhase(phi1 + phi2)]
    )
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


def test_combine_global_phases():
    """Test that the ``combine_global_phases`` function implements the expected transform on a
    QuantumScript and check the equivalence between statevectors before and after the transform."""
    transformed_qfunc = combine_global_phases(original_qfunc)

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = qml.QNode(transformed_qfunc, device=dev)

    phi1 = 1.23
    phi2 = 4.56
    expected_qscript = qml.tape.make_qscript(expected_qfunc)(phi1, phi2)
    transformed_qscript = qml.tape.make_qscript(transformed_qfunc)(phi1, phi2)

    original_state = original_qnode(phi1, phi2, return_state=True)
    transformed_state = transformed_qnode(phi1, phi2, return_state=True)

    # check the equivalence between expected and transformed quantum scripts
    qml.assert_equal(expected_qscript, transformed_qscript)

    # check the equivalence between statevectors before and after the transform
    assert np.allclose(original_state, transformed_state)


@pytest.mark.autograd
def test_differentiability_autograd():
    """Test that the output of the ``combine_global_phases`` transform is differentiable with autograd"""
    import pennylane.numpy as pnp

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = combine_global_phases(original_qnode)

    phi1 = pnp.array(0.25)
    phi2 = pnp.array(-0.6)
    grad1, grad2 = qml.jacobian(transformed_qnode)(phi1, phi2)

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)


@pytest.mark.jax
@pytest.mark.parametrize("use_jit", [False, True])
def test_differentiability_jax(use_jit):
    """Test that the output of the ``combine_global_phases`` transform is differentiable with JAX"""
    import jax
    import jax.numpy as jnp

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = combine_global_phases(original_qnode)

    if use_jit:
        transformed_qnode = jax.jit(transformed_qnode)

    phi1 = jnp.array(0.25)
    phi2 = jnp.array(-0.6)
    grad1, grad2 = jax.jacobian(transformed_qnode, argnums=[0, 1])(phi1, phi2)

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)


@pytest.mark.torch
def test_differentiability_torch():
    """Test that the output of the ``combine_global_phases`` transform is differentiable with Torch"""
    import torch
    from torch.autograd.functional import jacobian

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = combine_global_phases(original_qnode)

    phi1 = torch.tensor(0.25)
    phi2 = torch.tensor(-0.6)
    grad1, grad2 = jacobian(transformed_qnode, (phi1, phi2))

    zero = torch.tensor(0.0)
    assert qml.math.isclose(grad1, zero)
    assert qml.math.isclose(grad2, zero)


@pytest.mark.tf
def test_differentiability_tensorflow():
    """Test that the output of the ``combine_global_phases`` transform is differentiable with TensorFlow"""
    import tensorflow as tf

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)

    phi1 = tf.Variable(0.25)
    phi2 = tf.Variable(-0.6)
    with tf.GradientTape() as tape:
        transformed_qnode = combine_global_phases(original_qnode)(phi1, phi2)
    grad1, grad2 = tape.jacobian(transformed_qnode, (phi1, phi2))

    assert qml.math.isclose(grad1, 0.0)
    assert qml.math.isclose(grad2, 0.0)
