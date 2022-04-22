import pennylane.numpy as np
import pennylane as qml

from pennylane.devices.default_qubit import DefaultQubit


class CustomJacobianDevice(DefaultQubit):
    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities()
        capabilities["provides_jacobian"] = True
        return capabilities

    def jacobian(self, tape):
        return np.array([1.0, 2.0, 3.0, 4.0])


class TestCustomJacobian:
    def test_custom_jacobians(self):
        dev = CustomJacobianDevice(wires=2)

        @qml.qnode(dev, diff_method="device")
        def circuit(v):
            qml.RX(v, wires=0)
            return qml.probs(wires=[0, 1])

        d_circuit = qml.jacobian(circuit, argnum=0)

        params = np.array(1.0, requires_grad=True)

        d_out = d_circuit(params)
        assert np.allclose(d_out, np.array([1.0, 2.0, 3.0, 4.0]))
