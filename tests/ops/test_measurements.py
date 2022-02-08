import pytest
import math

import pennylane as qml
import pennylane.numpy as np

class TestMidCircuitMeasurements:
    """Tests the continuous variable based operations."""

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("r", np.linspace(0.0, 1.6, 10))
    def test_quantum_teleportation(self, r, device):
        dev = qml.device(device, wires=3)

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RY(rads, wires=0)

            return qml.probs(wires=0)

        @qml.qnode(dev)
        def teleportation_circuit(rads):

            # Create Alice's secret qubit state
            qml.RY(rads, wires=0)

            # create bell state between wires 1, 2. 1 is held by Alice and 2 held by Bob
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])

            # Alice sends her qubits through a CNOT gate.
            qml.CNOT(wires=[0, 1])

            # Alice then sends the first qubit through a Hadamard gate.
            qml.Hadamard(wires=0)

            # Alice measures her qubits, obtaining one of four results, and sends this information to Bob.
            m_0 = qml.Measure(0)
            m_1 = qml.Measure(1)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qml.If(m_1, qml.RX(math.pi, wires=2, do_queue=False))
            qml.If(m_0, qml.RZ(math.pi, wires=2, do_queue=False))

            return qml.probs(wires=2)

        normal_probs = normal_circuit(r)
        teleported_probs = teleportation_circuit(r)

        assert np.linalg.norm(normal_probs - teleported_probs) < 0.01
