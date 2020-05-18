import numpy as np
import pennylane as qml


np.random.seed(0)


def rx_ansatz(phis, **kwargs):
    for w, phi in enumerate(phis):
        qml.RX(phi, wires=w)


def layer_ansatz(weights, x=None, **kwargs):
    qml.templates.AngleEmbedding(x, wires=[0, 1, 2])
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])


class TestMSECost:
    def test_layer_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)

        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]

        cost = qml.qnn.MSECost(layer_ansatz, observables, dev)
        weights = np.random.rand(num_qubits, 3, 3)
        res = cost(weights, x=[1., 2., 1.], target=[1.1, 2., 1.1])

        assert np.allclose(res, np.array([1.7, 4.4, 1.3]), atol=0.1, rtol=0.1)

    def test_rx_circuit(self):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)

        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]

        cost = qml.qnn.MSECost(rx_ansatz, observables, dev)
        phis = np.random.rand(num_qubits)

        res = cost(phis, target=[1.1, 2., 1.1])

        assert np.allclose(res, np.array([0.2, 4., 0.]), atol=0.1, rtol=0.1)
