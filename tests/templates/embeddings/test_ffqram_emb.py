"""
Tests for the FFQRAMEmbedding template.
"""

import numpy as np

import pennylane as qml


# pylint: disable=protected-access
def test_flatten_unflatten():
    """Test the _flatten and _unflatten methods."""
    amplitudes = [3.0, 4.0]
    address = ["00", "11"]
    op = qml.FFQRAMEmbedding(amplitudes, wires=[0, 1, 2], address=address)

    data, metadata = op._flatten()
    assert data == op.data
    assert len(metadata) == 2
    assert metadata[0] == op.wires
    assert len(metadata[1]) == 1
    assert metadata[1][0][0] == "address"
    assert np.allclose(metadata[1][0][1], [[0, 0], [1, 1]])

    new_op = type(op)._unflatten(*op._flatten())
    qml.assert_equal(op, new_op)
    assert op is not new_op


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    def test_expansion(self):
        """Checks the queue for the default settings."""
        amplitudes = [3.0, 4.0]
        op = qml.FFQRAMEmbedding(amplitudes, wires=[0, 1, 2], address=["00", "11"])
        tape = qml.tape.QuantumScript(op.decomposition())

        expected_names = [
            "Hadamard",
            "Hadamard",
            "PauliX",
            "PauliX",
            "C(RY)",
            "PauliX",
            "PauliX",
            "C(RY)",
        ]
        expected_wires = [[0], [1], [0], [1], [0, 1, 2], [0], [1], [0, 1, 2]]

        assert [gate.name for gate in tape.operations] == expected_names
        assert [gate.wires.tolist() for gate in tape.operations] == expected_wires

        expected_angles = 2 * np.arcsin(np.array(amplitudes) / np.linalg.norm(amplitudes))
        assert np.allclose(tape.operations[4].parameters, [expected_angles[0]])
        assert np.allclose(tape.operations[7].parameters, [expected_angles[1]])
