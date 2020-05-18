import numpy as np
import pennylane as qml
import pytest


def rx_ansatz(phis, **kwargs):
    for w, phi in enumerate(phis):
        qml.RX(phi, wires=w)


def layer_ansatz(weights, x=None, **kwargs):
    qml.templates.AngleEmbedding(x, wires=[0, 1, 2])
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])


class TestMSECost:
    @pytest.mark.parametrize("interface", qml.qnodes.decorator.ALLOWED_INTERFACES)
    def test_layer_circuit(self, interface):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)

        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]

        cost = qml.qnn.MSECost(layer_ansatz, observables, dev, interface=interface)
        weights = np.ones((num_qubits, 3, 3))
        res = cost(weights, x=np.array([1., 2., 1.]), target=np.array([1.1, 2., 1.1]))

        assert np.allclose(res, np.array([1.0, 5.8, 1.5]), atol=0.1, rtol=0.1)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)

        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]

        cost = qml.qnn.MSECost(rx_ansatz, observables, dev)
        phis = np.ones(num_qubits)

        res = cost(phis, target=[1.1, 2., 1.1])

        assert np.allclose(res, np.array([0.3, 4., 0.6]), atol=0.1, rtol=0.1)
