"""
Tests for the combine_global_phases transform.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import combine_global_phases


def original_qfunc(phi1, phi2):
    qml.Hadamard(wires=1)
    qml.GlobalPhase(phi1, wires=[0, 1])
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.GlobalPhase(phi2, wires=1)
    qml.CNOT(wires=[2, 0])
    return qml.state()


def expected_qfunc(phi1, phi2):
    qml.Hadamard(wires=1)
    qml.PauliY(wires=0)
    qml.PauliX(wires=2)
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 0])
    qml.GlobalPhase(phi1 + phi2)
    return qml.state()


def test_no_global_phase_gate():
    """Test that when the input ``QuantumScript`` has no ``qml.GlobalPhase`` gate, the returned output is exactly the same"""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)])

    expected_qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)])
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


@pytest.mark.parametrize("phi", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_single_global_phase_gate(phi):
    """Test that when the input ``QuantumScript`` has a single ``qml.GlobalPhase`` gate, the returned output has an equivalent
    ``qml.GlobalPhase`` operation appended at the end"""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.GlobalPhase(phi, 0), qml.RX(0, 0)])

    expected_qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0), qml.GlobalPhase(phi)])
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_multiple_global_phase_gates(phi1, phi2):
    """Test that when the input ``QuantumScript`` has multiple ``qml.GlobalPhase`` gates, the returned output has an equivalent
    single ``qml.GlobalPhase`` operation appended at the end with a total phase being equal to the sum of each original global phase
    """
    qscript = qml.tape.QuantumScript(
        [qml.GlobalPhase(phi1, 0), qml.Hadamard(0), qml.GlobalPhase(phi2, 0), qml.RX(0, 0)]
    )

    expected_qscript = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.RX(0, 0), qml.GlobalPhase(phi1 + phi2)]
    )
    (transformed_qscript,), _ = combine_global_phases(qscript)

    qml.assert_equal(expected_qscript, transformed_qscript)


@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_combine_global_phases(phi1, phi2):
    """Test that the transform works in the autograd interface"""
    transformed_qfunc = combine_global_phases(original_qfunc)

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)
    transformed_qnode = qml.QNode(transformed_qfunc, device=dev)

    expected_qscript = qml.tape.make_qscript(expected_qfunc)(phi1, phi2)
    transformed_qscript = qml.tape.make_qscript(transformed_qfunc)(phi1, phi2)

    original_state = original_qnode(phi1, phi2)
    transformed_state = transformed_qnode(phi1, phi2)

    # check the equivalence between expected and transformed quantum scripts
    qml.assert_equal(expected_qscript, transformed_qscript)

    # check the equivalence between statevectors before and after the transform
    assert np.allclose(original_state, transformed_state)


@pytest.mark.jax
@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_combine_global_phases_jax(phi1, phi2):
    """Test that the transform works in the JAX interface"""
    dev = qml.device("default.qubit", wires=3)

    original_qnode = qml.QNode(original_qfunc, device=dev, interface="jax")
    transformed_qnode = combine_global_phases(original_qnode)

    original_state = original_qnode(phi1, phi2)
    transformed_state = transformed_qnode(phi1, phi2)

    # check the equivalence between statevectors before and after the transform
    assert np.allclose(original_state, transformed_state)


@pytest.mark.torch
@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_combine_global_phases_torch(phi1, phi2):
    """Test that the transform works in the Torch interface"""
    dev = qml.device("default.qubit", wires=3)

    original_qnode = qml.QNode(original_qfunc, device=dev, interface="torch")
    transformed_qnode = combine_global_phases(original_qnode)

    original_state = original_qnode(phi1, phi2)
    transformed_state = transformed_qnode(phi1, phi2)

    # check the equivalence between statevectors before and after the transform
    assert np.allclose(original_state, transformed_state)


@pytest.mark.tf
@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_combine_global_phases_tf(phi1, phi2):
    """Test that the transform works in the TensorFlow interface"""
    dev = qml.device("default.qubit", wires=3)

    original_qnode = qml.QNode(original_qfunc, device=dev, interface="tensorflow")
    transformed_qnode = combine_global_phases(original_qnode)

    original_state = original_qnode(phi1, phi2)
    transformed_state = transformed_qnode(phi1, phi2)

    # check the equivalence between statevectors before and after the transform
    assert np.allclose(original_state, transformed_state)
