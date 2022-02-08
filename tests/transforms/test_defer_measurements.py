import pytest
import math

import pennylane as qml
import pennylane.numpy as np

class TestMidCircuitMeasurements:
    """Tests mid circuit measurements"""

    @pytest.mark.parametrize("r", np.linspace(0.0, 1.6, 10))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    def test_quantum_teleportation(self, device, r):
        dev = qml.device(device, wires=3)

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RY(rads, wires=0)

            return qml.probs(wires=0)

        @qml.qnode(dev)
        @qml.defer_measurements
        def teleportation_circuit(rads):

            # Create Alice's secret qubit state
            qml.RY(rads, wires=0)

            # create an EPR pair with wires 1 and 2. 1 is held by Alice and 2 held by Bob
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])

            # Alice sends her qubits through a CNOT gate.
            qml.CNOT(wires=[0, 1])

            # Alice then sends the first qubit through a Hadamard gate.
            qml.Hadamard(wires=0)

            # Alice measures her qubits, obtaining one of four results, and sends this information to Bob.
            m_0 = qml.mid_measure(0)
            m_1 = qml.mid_measure(1)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qml.if_then(m_1, qml.RX)(math.pi, wires=2)
            qml.if_then(m_0, qml.RZ)(math.pi, wires=2)

            return qml.probs(wires=2)

        normal_probs = normal_circuit(r)
        teleported_probs = teleportation_circuit(r)

        assert np.linalg.norm(normal_probs - teleported_probs) < 0.01
