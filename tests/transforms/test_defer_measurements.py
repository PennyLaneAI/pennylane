import pytest
import math

import pennylane as qml
import pennylane.numpy as np

class TestQNode:
    """Test that the transform integrates well with QNodes."""

    def test_only_mcm(self):
        """Test that a quantum function that only contains one mid-circuit
        measurement yields the correct results and is transformed correctly."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode1():
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2():
            m = qml.mid_measure(1)
            return qml.expval(qml.PauliZ(0))

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        for op1, op2 in zip(qnode1.qtape.queue, qnode2.qtape.queue):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

    def test_ops_before_after(self):
        """Test that a quantum function that contains one operation before and
        after a mid-circuit measurement yields the correct results and is
        transformed correctly."""
        dev = qml.device("default.qubit", wires=3)

        def func1():
            qml.RY(0.123, wires=0)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        def func2():
            qml.RY(0.123, wires=0)
            qml.mid_measure(1)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        tape_deferred_func = qml.defer_measurements(func2)
        qnode1 = qml.QNode(func1, dev)
        qnode2 = qml.QNode(tape_deferred_func, dev)

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        for op1, op2 in zip(qnode1.qtape.queue, qnode2.qtape.queue):
            assert type(op1) == type(op2)
            assert op1.data == op2.data


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

        assert np.allclose(normal_probs, teleported_probs)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("ops", [(qml.RX, qml.CRX), (qml.RY, qml.CRY), (qml.RZ, qml.CRZ)])
    def test_conditional_rotations(self, device, r, ops):
        dev = qml.device(device, wires=3)

        op, controlled_op = ops

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qml.probs(wires=1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def teleportation_circuit(rads):
            qml.Hadamard(0)
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, op)(rads, wires=1)
            return qml.probs(wires=1)

        normal_probs = normal_circuit(r)
        teleported_probs = teleportation_circuit(r)

        assert np.allclose(normal_probs, teleported_probs)
