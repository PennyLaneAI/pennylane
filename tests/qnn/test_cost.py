import numpy as np
import pennylane as qml
import pytest


def rx_ansatz(phis, **kwargs):
    for w, phi in enumerate(phis):
        qml.RX(phi, wires=w)


def layer_ansatz(weights, x=None, **kwargs):
    qml.templates.AngleEmbedding(x, wires=[0, 1, 2])
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])


@pytest.mark.parametrize("interface", qml.qnodes.decorator.ALLOWED_INTERFACES)
@pytest.mark.usefixtures("skip_if_no_torch_support", "skip_if_no_tf_support")
class TestMSECost:
    def test_no_target(self, interface):
        with pytest.raises(ValueError, match="The target cannot be None"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            cost = qml.qnn.MSECost(rx_ansatz, observables, dev, interface=interface)

            phis = np.ones(num_qubits)
            cost(phis)

    def test_invalid_target(self, interface):
        with pytest.raises(ValueError, match="Invalid target"):
            num_qubits = 1

            dev = qml.device("default.qubit", wires=num_qubits)
            observables = [qml.PauliZ(0)]
            cost = qml.qnn.MSECost(rx_ansatz, observables, dev, interface=interface)

            phis = np.ones(num_qubits)
            cost(phis, target=np.array([1.0, 2.0]))

    def test_layer_circuit(self, interface):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        cost = qml.qnn.MSECost(layer_ansatz, observables, dev, interface=interface)

        weights = np.ones((num_qubits, 3, 3))
        res = cost(weights, x=np.array([1.0, 2.0, 1.0]), target=np.array([1.1, 2.0, 1.1]))

        assert np.allclose(res, np.array([1.0, 5.8, 1.5]), atol=0.1, rtol=0.1)

    def test_rx_circuit(self, interface):
        num_qubits = 3

        dev = qml.device("default.qubit", wires=num_qubits)
        observables = [qml.PauliZ(0), qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]
        cost = qml.qnn.MSECost(rx_ansatz, observables, dev, interface=interface)

        phis = np.ones(num_qubits)
        res = cost(phis, target=np.array([1.1, 2.0, 1.1]))

        assert np.allclose(res, np.array([0.3, 4.0, 0.6]), atol=0.1, rtol=0.1)
