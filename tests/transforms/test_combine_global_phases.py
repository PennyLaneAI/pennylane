"""
Tests for the combine_global_phases transform.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import combine_global_phases


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
def test_output_state(phi1, phi2):
    """Test if the statevector returned by the transformed circuit is equivalent to the statevector returned by the original circuit"""

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(device=dev)
    def original_circuit():
        qml.Hadamard(wires=1)
        qml.GlobalPhase(phi1, wires=[0, 1])
        qml.PauliY(wires=0)
        qml.PauliX(wires=2)
        qml.CNOT(wires=[1, 2])
        qml.GlobalPhase(phi2, wires=1)
        qml.CNOT(wires=[2, 0])
        return qml.state()

    transformed_circuit = combine_global_phases(original_circuit)

    original_state = original_circuit()
    transformed_state = transformed_circuit()
    assert np.allclose(original_state, transformed_state)
