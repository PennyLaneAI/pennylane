# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the Fourier spectrum transform.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.circuit_spectrum import circuit_spectrum


class TestCircuits:
    """Tests that the spectrum is returned as expected."""

    @pytest.mark.parametrize("n_layers, n_qubits", [(1, 1), (2, 3), (4, 1)])
    def test_spectrum_grows_with_gates(self, n_layers, n_qubits):
        """Test that the spectrum grows linearly with the number of
        encoding gates if we use Pauli rotation encoding."""

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x):
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(x, wires=i, id="x")
                    qml.RY(0.4, wires=i)
            return qml.expval(qml.PauliZ(wires=0))

        res = circuit_spectrum(circuit)(0.1)
        expected_degree = n_qubits * n_layers
        assert np.allclose(res["x"], range(-expected_degree, expected_degree + 1))

    def test_encoding_gates(self):
        """Test that the spectrum contains the ids provided in encoding_gates, or
        all ids if encoding_gates is None."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0, id="x")
            qml.RY(0.4, wires=0, id="other")
            return qml.expval(qml.PauliZ(wires=0))

        res = circuit_spectrum(circuit, encoding_gates=["x"])(0.1)
        assert res == {"x": [-1.0, 0.0, 1.0]}

        res = circuit_spectrum(circuit, encoding_gates=["x", "other"])(0.1)
        assert res == {"x": [-1.0, 0.0, 1.0], "other": [-1.0, 0.0, 1.0]}

        res = circuit_spectrum(circuit)(0.1)
        assert res == {"x": [-1.0, 0.0, 1.0], "other": [-1.0, 0.0, 1.0]}

        res = circuit_spectrum(circuit, encoding_gates=["a"])(0.1)
        assert res == {"a": []}

    def test_spectrum_changes_with_qnode_args(self):
        """Test that the spectrum changes per call if a qnode argument changes the
        circuit architecture."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(last_gate):
            qml.RX(0.1, wires=0, id="x")
            qml.RX(0.2, wires=1, id="x")
            if last_gate:
                qml.RX(0.3, wires=2, id="x")
            return qml.expval(qml.PauliZ(wires=0))

        res_true = circuit_spectrum(circuit)(True)
        assert np.allclose(res_true["x"], range(-3, 4))

        res_false = circuit_spectrum(circuit)(False)
        assert np.allclose(res_false["x"], range(-2, 3))

    def test_input_gates_not_of_correct_form(self):
        """Test that an error is thrown if gates marked as encoding gates
        are not single-parameter gates."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.1, wires=0, id="x")
            qml.Rot(0.2, 0.3, 0.4, wires=1, id="x")
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.raises(ValueError, match="Can only consider one-parameter gates"):
            circuit_spectrum(circuit)()


def circuit(x, w):
    """Test circuit"""
    for l in range(2):
        for i in range(3):
            qml.RX(x[i], wires=0, id="x" + str(i))
            qml.RY(w[l][i], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
    qml.RZ(x[0], wires=0, id="x0")
    return qml.expval(qml.PauliZ(wires=0))


expected_result = {
    "x0": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "x1": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "x2": [-2.0, -1.0, 0.0, 1.0, 2.0],
}


class TestInterfaces:
    """Test that inputs are correctly identified and spectra computed in
    all interfaces."""

    @pytest.mark.autograd
    def test_integration_autograd(self):
        """Test that the spectra of a circuit is calculated correctly
        in the autograd interface."""

        x = pnp.array([1.0, 2.0, 3.0], requires_grad=False)
        w = pnp.array([[-1, -2, -3], [-4, -5, -6]], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="autograd")

        res = circuit_spectrum(qnode)(x, w)
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    @pytest.mark.torch
    def test_integration_torch(self):
        """Test that the spectra of a circuit is calculated correctly
        in the torch interface."""

        import torch

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        w = torch.tensor([[-1, -2, -3], [-4, -5, -6]], requires_grad=False)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="torch")

        res = circuit_spectrum(qnode)(x, w)
        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    @pytest.mark.tf
    def test_integration_tf(self):
        """Test that the spectra of a circuit is calculated correctly
        in the tf interface."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="tf")

        x = tf.Variable([1.0, 2.0, 3.0])
        w = tf.constant([[-1, -2, -3], [-4, -5, -6]])
        res = circuit_spectrum(qnode)(x, w)

        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    @pytest.mark.jax
    def test_integration_jax(self):
        """Test that the spectra of a circuit is calculated correctly
        in the jax interface."""
        from jax import numpy as jnp

        x = jnp.array([1.0, 2.0, 3.0])
        w = [[-1, -2, -3], [-4, -5, -6]]

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="jax")

        res = circuit_spectrum(qnode)(x, w)

        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2
