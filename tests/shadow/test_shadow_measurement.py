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
"""Unit tests for the classical shadows measurement process"""
# pylint:disable=no-self-use, import-outside-toplevel, redefined-outer-name, unpacking-non-sequence, too-few-public-methods, not-an-iterable, inconsistent-return-statements
import copy
import pytest

import pennylane as qml
from pennylane import numpy as np


def get_circuit(wires, shots, seed_recipes, interface="autograd", device="default.qubit"):
    """
    Return a QNode that prepares the state (|00...0> + |11...1>) / sqrt(2)
        and performs the classical shadow measurement
    """
    if device is not None:
        dev = qml.device(device, wires=wires, shots=shots)
    else:
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        # make the device call the superclass method to switch between the general qubit device and device specific implementations (i.e. for default qubit)
        dev.classical_shadow = super(type(dev), dev).classical_shadow

    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)

        for target in range(1, wires):
            qml.CNOT(wires=[0, target])

        return qml.classical_shadow(wires=range(wires), seed_recipes=seed_recipes)

    return circuit


def get_x_basis_circuit(wires, shots, interface="autograd", device="default.qubit"):
    """
    Return a QNode that prepares the |++..+> state and performs a classical shadow measurement
    """
    if device is not None:
        dev = qml.device(device, wires=wires, shots=shots)
    else:
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        # make the device call the superclass method to switch between the general qubit device and device specific implementations (i.e. for default qubit)
        dev.classical_shadow = super(type(dev), dev).classical_shadow

    @qml.qnode(dev, interface=interface)
    def circuit():
        for wire in range(wires):
            qml.Hadamard(wire)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def get_y_basis_circuit(wires, shots, interface="autograd", device="default.qubit"):
    """
    Return a QNode that prepares the |+i>|+i>...|+i> state and performs a classical shadow measurement
    """
    if device is not None:
        dev = qml.device(device, wires=wires, shots=shots)
    else:
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        # make the device call the superclass method to switch between the general qubit device and device specific implementations (i.e. for default qubit)
        dev.classical_shadow = super(type(dev), dev).classical_shadow

    @qml.qnode(dev, interface=interface)
    def circuit():
        for wire in range(wires):
            qml.Hadamard(wire)
            qml.RZ(np.pi / 2, wire)
        return qml.classical_shadow(wires=range(wires))

    return circuit


def get_z_basis_circuit(wires, shots, interface="autograd", device="default.qubit"):
    """
    Return a QNode that prepares the |00..0> state and performs a classical shadow measurement
    """
    if device is not None:
        dev = qml.device(device, wires=wires, shots=shots)
    else:
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        # make the device call the superclass method to switch between the general qubit device and device specific implementations (i.e. for default qubit)
        dev.classical_shadow = super(type(dev), dev).classical_shadow

    @qml.qnode(dev, interface=interface)
    def circuit():
        return qml.classical_shadow(wires=range(wires))

    return circuit


wires_list = [1, 3]


@pytest.mark.parametrize("wires", wires_list)
class TestShadowMeasurement:
    """Unit tests for classical_shadow measurement"""

    shots_list = [1, 100]
    seed_recipes_list = [True, False]

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_numeric_type(self, wires, seed):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        res = qml.classical_shadow(wires=range(wires), seed_recipes=seed)
        assert res.numeric_type == int

    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_shape(self, wires, shots, seed):
        """Test that the shape of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        res = qml.classical_shadow(wires=range(wires), seed_recipes=seed)
        assert res.shape(device=dev) == (1, 2, shots, wires)

        # test an error is raised when device is None
        msg = "The device argument is required to obtain the shape of a classical shadow measurement process"
        with pytest.raises(qml.measurements.MeasurementShapeError, match=msg):
            res.shape(device=None)

    def test_shape_matches(self, wires):
        """Test that the shape of the MeasurementProcess matches the shape
        of the tape execution"""
        shots = 100

        circuit = get_circuit(wires, shots, True)
        circuit.construct((), {})

        res = qml.execute([circuit.tape], circuit.device, None)[0]
        expected_shape = qml.classical_shadow(wires=range(wires)).shape(device=circuit.device)

        assert res.shape == expected_shape

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_copy(self, wires, seed):
        """Test that the attributes of the MeasurementProcess instance are
        correctly copied"""
        res = qml.classical_shadow(wires=range(wires), seed_recipes=seed)

        copied_res = copy.copy(res)
        assert type(copied_res) == type(res)
        assert copied_res.return_type == res.return_type
        assert copied_res.wires == res.wires
        assert copied_res.seed == res.seed

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("seed", seed_recipes_list)
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", None])
    def test_format(self, wires, shots, seed, interface, device):
        """Test that the format of the returned classical shadow
        measurement is correct"""
        import tensorflow as tf
        import torch

        circuit = get_circuit(wires, shots, seed, interface, device)
        shadow = circuit()

        # test shape is correct
        assert shadow.shape == (2, shots, wires)

        # test dtype is correct
        expected_dtype = np.int8
        if interface == "tf":
            expected_dtype = tf.int8
        elif interface == "torch":
            expected_dtype = torch.int8

        assert shadow.dtype == expected_dtype

        bits, recipes = shadow

        # test allowed values of bits and recipes
        assert qml.math.all(np.logical_or(bits == 0, bits == 1))
        assert qml.math.all(np.logical_or(recipes == 0, np.logical_or(recipes == 1, recipes == 2)))

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize("device", ["default.qubit", None])
    @pytest.mark.parametrize(
        "circuit_fn, basis_recipe",
        [(get_x_basis_circuit, 0), (get_y_basis_circuit, 1), (get_z_basis_circuit, 2)],
    )
    def test_return_distribution(self, wires, interface, device, circuit_fn, basis_recipe):
        """Test that the distribution of the bits and recipes are correct for a circuit
        that prepares all qubits in a Pauli basis"""
        # high number of shots to prevent true negatives
        shots = 1000

        circuit = circuit_fn(wires, shots=shots, interface=interface, device=device)
        bits, recipes = circuit()

        # test that the recipes follow a rough uniform distribution
        ratios = np.unique(recipes, return_counts=True)[1] / (wires * shots)
        assert np.allclose(ratios, 1 / 3, atol=1e-1)

        # test that the bit is 0 for all X measurements
        assert qml.math.allequal(bits[recipes == basis_recipe], 0)

        # test that the bits are uniformly distributed for all Y and Z measurements
        bits1 = bits[recipes == (basis_recipe + 1) % 3]
        ratios1 = np.unique(bits1, return_counts=True)[1] / bits1.shape[0]
        assert np.allclose(ratios1, 1 / 2, atol=1e-1)

        bits2 = bits[recipes == (basis_recipe + 2) % 3]
        ratios2 = np.unique(bits2, return_counts=True)[1] / bits2.shape[0]
        assert np.allclose(ratios2, 1 / 2, atol=1e-1)

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_shots_none_error(self, wires, seed):
        """Test that an error is raised when a device with shots=None is used
        to obtain classical shadows"""
        circuit = get_circuit(wires, None, seed)

        msg = "The number of shots has to be explicitly set on the device when using sample-based measurements"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit()

    @pytest.mark.parametrize("shots", shots_list)
    def test_multi_measurement_error(self, wires, shots):
        """Test that an error is raised when classical shadows is returned
        with other measurement processes"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)

            for target in range(1, wires):
                qml.CNOT(wires=[0, target])

            return qml.classical_shadow(wires=range(wires)), qml.expval(qml.PauliZ(0))

        msg = "Classical shadows cannot be returned in combination with other return types"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            shadow = circuit()
