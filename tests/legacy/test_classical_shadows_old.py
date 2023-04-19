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
"""Unit tests for the classical shadows measurement processes"""

import copy

import autograd.numpy
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import ClassicalShadowMP, Shots
from pennylane.measurements.classical_shadow import ShadowExpvalMP


def hadamard_circuit(wires, shots=10000, interface="autograd"):
    dev = qml.device("default.qubit", wires=wires, shots=shots)

    @qml.qnode(dev, interface=interface)
    def circuit(obs, k=1):
        for i in range(wires):
            qml.Hadamard(wires=i)
        return qml.shadow_expval(obs, k=k)

    return circuit


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

        return qml.classical_shadow(wires=range(wires), seed=seed_recipes)

    return circuit


wires_list = [1, 3]


@pytest.mark.parametrize("wires", wires_list)
class TestClassicalShadow:
    """Unit tests for classical_shadow measurement"""

    shots_list = [1, 100]
    seed_recipes_list = [None, 74]  # random seed

    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_numeric_type(self, wires, seed):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        res = qml.classical_shadow(wires=range(wires), seed=seed)
        assert res.numeric_type == int

    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("seed", seed_recipes_list)
    def test_measurement_process_shape(self, wires, shots, seed):
        """Test that the shape of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        num_shots = Shots(shots)
        res = qml.classical_shadow(wires=range(wires), seed=seed)
        assert res.shape(dev, num_shots) == (1, 2, shots, wires)

        # test an error is raised when device is None
        msg = "Shots must be specified to obtain the shape of a classical shadow measurement"
        with pytest.raises(qml.measurements.MeasurementShapeError, match=msg):
            res.shape(dev, Shots(None))

    def test_shape_matches(self, wires):
        """Test that the shape of the MeasurementProcess matches the shape
        of the tape execution"""
        shots = 100

        circuit = get_circuit(wires, shots, True)
        circuit.construct((), {})

        res = qml.execute([circuit.tape], circuit.device, None)[0]
        expected_shape = qml.classical_shadow(wires=range(wires)).shape(
            circuit.device, Shots(shots)
        )

        assert res.shape == expected_shape


@pytest.mark.autograd
class TestExpvalMeasurement:
    def test_measurement_process_numeric_type(self):
        """Test that the numeric type of the MeasurementProcess instance is correct"""
        H = qml.PauliZ(0)
        res = qml.shadow_expval(H)
        assert res.numeric_type == float

    @pytest.mark.parametrize("wires", [1, 2])
    @pytest.mark.parametrize("shots", [1, 10])
    def test_measurement_process_shape(self, wires, shots):
        """Test that the shape of the MeasurementProcess instance is correct"""
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        H = qml.PauliZ(0)
        res = qml.shadow_expval(H)
        assert res.shape(dev, Shots(shots)) == (1,)

    def test_shape_matches(self):
        """Test that the shape of the MeasurementProcess matches the shape
        of the tape execution"""
        wires = 2
        shots = 100
        H = qml.PauliZ(0)

        circuit = hadamard_circuit(wires, shots)
        circuit.construct((H,), {})

        res = qml.execute([circuit.tape], circuit.device, None)[0]
        expected_shape = qml.shadow_expval(H).shape(circuit.device, Shots(shots))

        assert res.shape == expected_shape
