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


@pytest.mark.parametrize(
    "ml_interface",
    [
        pytest.param("autograd", marks=pytest.mark.autograd),
        pytest.param("jax", marks=pytest.mark.jax),
        pytest.param("torch", marks=pytest.mark.torch),
        pytest.param("tensorflow", marks=pytest.mark.tf),
    ],
)
def test_differentiability(ml_interface):
    """Test that differentiability holds before and after the transform."""
    phi1 = qml.math.asarray(0.25, like=ml_interface)
    phi2 = qml.math.asarray(-0.6, like=ml_interface)

    dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(original_qfunc, device=dev)

    jac = qml.math.jacobian(original_qnode)(phi1, phi2)
    assert not jac

    transformed_qnode = remove_global_phases(original_qnode)

    jac = qml.math.jacobian(transformed_qnode)(phi1, phi2)
    assert not jac
