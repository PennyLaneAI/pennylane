"""
Tests for the remove_global_phases transform.
"""

import pytest

import pennylane as qml
from pennylane.transforms import remove_global_phases


@pytest.mark.parametrize("measurements", ([qml.state()], [qml.state(), qml.probs()]))
def test_global_phase_removal_error(measurements):
    """Test that an error is raised if the transform is applied to a circuit with no measurements."""
    qscript = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)], measurements)

    with pytest.raises(
        qml.QuantumFunctionError, match="The quantum circuit cannot contain a state measurement"
    ):
        remove_global_phases(qscript)


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

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(weights):
        qml.GlobalPhase(weights[0, 0, 0])
        qml.RX(weights[0, 0, 0], wires=0)
        qml.RY(weights[0, 0, 1], wires=1)
        qml.GlobalPhase(weights[1, 0, 2])
        qml.RZ(weights[1, 0, 2], wires=0)
        return qml.probs()

    weights = qml.math.asarray(
        [[[0.2, 0.9, -1.4]], [[0.5, 0.2, 0.1]]], like=ml_interface, requires_grad=True
    )

    jac1 = qml.math.jacobian(circuit)(weights)
    assert jac1.shape == (4, 2, 1, 3)

    transformed_qnode = remove_global_phases(circuit)

    jac2 = qml.math.jacobian(transformed_qnode)(weights)
    assert jac2.shape == (4, 2, 1, 3)

    qml.math.allclose(jac1, jac2)
