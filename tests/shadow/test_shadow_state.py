# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the classical shadows state measurement process"""

import pytest
import copy

import pennylane as qml
import pennylane.numpy as np
from pennylane.shadows import ClassicalShadow


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.classical_shadow_state(wires=range(wires))

    return circuit


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow_state(wires=range(wires))

    return circuit


def qft_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    one_state = np.zeros(wires)
    one_state[-1] = 1

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.BasisState(one_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.classical_shadow_state(wires=range(wires))

    return circuit


def strongly_entangling_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return qml.classical_shadow_state(wires=range(wires))

    return circuit


def strongly_entangling_circuit_exact(wires, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev, interface=interface)
    def circuit(x):
        qml.StronglyEntanglingLayers(weights=x, wires=range(wires))
        return qml.density_matrix(wires=range(wires))

    return circuit


@pytest.mark.autograd
class TestStateMeasurement:
    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_numeric_type(self, wires, shots):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        res = qml.classical_shadow_state(wires=range(wires))
        assert res.numeric_type == complex

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_shape(self, wires, shots):
        """Test that the shape of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        res = qml.classical_shadow_state(wires=range(wires))
        assert res.shape() == (2**wires, 2**wires)
        assert res.shape(dev) == (2**wires, 2**wires)

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_measurement_process_copy(self, wires, shots):
        """Test that the attributes of the MeasurementProcess instance are
        correctly copied"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        res = qml.classical_shadow_state(wires=range(wires))

        copied_res = copy.copy(res)
        assert type(copied_res) == type(res)
        assert copied_res.return_type == res.return_type
        assert copied_res.wires == res.wires
        assert copied_res.seed == res.seed

    @pytest.mark.parametrize("wires", [1, 2, 3])
    def test_shots_none_error(self, wires):
        """Test that an error is raised when a device with shots=None is used
        to obtain classical shadows"""
        circuit = hadamard_circuit(wires, None)

        msg = "The number of shots has to be explicitly set on the device when using sample-based measurements"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit()

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("shots", [1, 10, 100])
    def test_multi_measurement_error(self, wires, shots):
        """Test that an error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)

            for target in range(1, wires):
                qml.CNOT(wires=[0, target])

            return qml.classical_shadow_state(wires=range(wires)), qml.expval(qml.PauliZ(0))

        msg = "Classical shadows cannot be returned in combination with other return types"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit()


@pytest.mark.all_interfaces
class TestStateForward:
    """Test the classical_shadow_state measurement process forward pass"""

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_hadamard_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a uniform
        superposition of qubits"""
        import torch

        circuit = hadamard_circuit(wires, interface=interface)
        actual = circuit()

        assert actual.shape == (2**wires, 2**wires)
        assert actual.dtype == torch.complex128 if interface == "torch" else np.complex128

        expected = np.ones((2**wires, 2**wires)) / (2**wires)
        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_max_entangled_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a maximally
        entangled state"""
        import torch

        circuit = max_entangled_circuit(wires, interface=interface)
        actual = circuit()

        assert actual.shape == (2**wires, 2**wires)
        assert actual.dtype == torch.complex128 if interface == "torch" else np.complex128

        expected = np.zeros((2**wires, 2**wires))
        expected[np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1])] = 0.5

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_qft_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for a QFT state"""
        import torch

        circuit = qft_circuit(wires, interface=interface)
        actual = circuit()

        assert actual.shape == (2**wires, 2**wires)
        assert actual.dtype == torch.complex128 if interface == "torch" else np.complex128

        expected = np.exp(np.arange(2**wires) * 2j * np.pi / (2**wires)) / (2 ** (wires / 2))
        expected = np.outer(expected, np.conj(expected))

        assert qml.math.allclose(actual, expected, atol=1e-1)

    @pytest.mark.parametrize("wires", [1, 2, 3])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_subset_reconstruction(self, wires, interface):
        """Test that the state reconstruction is correct for partial wires"""
        import torch

        dev = qml.device("default.qubit", wires=4, shots=10000)

        @qml.qnode(dev, interface=interface)
        def circuit(measure_wires):
            qml.Hadamard(wires=0)
            for i in range(1, 4):
                qml.CNOT(wires=[0, i])
            return qml.classical_shadow_state(wires=range(measure_wires))

        actual = circuit(wires)
        assert actual.shape == (2**wires, 2**wires)
        assert actual.dtype == torch.complex128 if interface == "torch" else np.complex128

        expected = np.zeros((2**wires, 2**wires))
        expected[np.array([0, -1]), np.array([0, -1])] = 0.5
        assert qml.math.allclose(actual, expected, atol=1e-1)

    def test_large_state_warning(self, monkeypatch):
        """Test that a warning is raised when a very large state is reconstructed"""
        circuit = hadamard_circuit(17, shots=2)

        msg = "Querying density matrices for n_wires > 16 is not recommended, operation will take a long time"

        with monkeypatch.context() as m:
            # don't run the actual state computation since we only want the warning
            m.setattr(
                ClassicalShadow,
                "_obtain_global_snapshots",
                lambda *args, **kwargs: np.array([1, 2, 3]),
            )

            with pytest.warns(UserWarning, match=msg):
                circuit()


class TestExpvalBackward:
    """Test the classical_shadow_state measurement process backward pass"""

    @pytest.mark.autograd
    def test_backward_autograd(self):
        """Test that the gradient of the state reconstruction is correct for
        the autograd interface"""
