"""
Tests for the Combine Global Phases transform.
"""

import numpy as np
import pytest

import pennylane as qml


@pytest.mark.parametrize("phi1", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
@pytest.mark.parametrize("phi2", [-2 * np.pi, -np.pi, -1, 0, 1, np.pi, 2 * np.pi])
def test_output_state(phi1, phi2):
    """Test if the statevector returned by the transformed circuit is equivalent to the statevector returned by the original circuit"""

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(device=dev)
    def original_circuit(phi1, phi2):
        qml.Hadamard(wires=1)
        qml.GlobalPhase(phi1, wires=[0, 1])
        qml.PauliY(wires=0)
        qml.PauliX(wires=2)
        qml.CNOT(wires=[1, 2])
        qml.GlobalPhase(phi2, wires=1)
        qml.CNOT(wires=[2, 0])
        return qml.state()

    @qml.qnode(device=dev)
    def transformed_circuit(phi):
        qml.Hadamard(wires=1)
        qml.PauliY(wires=0)
        qml.PauliX(wires=2)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])
        qml.GlobalPhase(phi)
        return qml.state()

    original_state = original_circuit(phi1, phi2)
    transformed_state = transformed_circuit(phi1 + phi2)
    assert np.allclose(original_state, transformed_state)
