# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`fourier` coefficient and spectra calculations.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np

from pennylane.fourier.coefficients import frequency_spectra, fourier_coefficients

dev_1 = qml.device("default.qubit", wires=1)
dev_2 = qml.device("default.qubit", wires=2)
dev_3 = qml.device("default.qubit", wires=3)


@qml.qnode(dev_1)
def circuit_one_qubit_one_param(inpt):
    qml.RX(inpt[0], wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_1)
def circuit_one_qubit_two_params(inpt):
    qml.RX(inpt[0], wires=0)
    qml.RY(inpt[1], wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_2)
def circuit_two_qubits_repeated_param(inpt):
    qml.RX(inpt[0], wires=0)
    qml.RY(inpt[0], wires=1)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_2)
def circuit_two_qubits_two_params(inpt):
    qml.RX(inpt[0], wires=0)
    qml.RY(inpt[1], wires=1)
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_2)
def circuit_two_qubits_controlled_rot(inpt):
    qml.Hadamard(wires=0)
    qml.CRY(inpt[0], wires=[0, 1])
    return qml.expval(qml.PauliZ(1))


class TestFourierSpectra:
    """Basic usage and edge-case tests for the measurement optimization utility functions."""

    @pytest.mark.parametrize(
        "circuit,inpt",
        [
            (circuit_one_qubit_one_param, np.array([0.1])),
            (circuit_one_qubit_two_params, np.array([0.1, 0.3])),
            (circuit_two_qubits_two_params, np.array([-0.4, 0.8])),
            (circuit_two_qubits_controlled_rot, np.array([1.34])),
        ],
    )
    def test_no_input_fourier_spectra(self, circuit, inpt):
        circuit(inpt)
        assert frequency_spectra(circuit.qtape) == {}

    @pytest.mark.parametrize(
        "circuit,inpt,spectra",
        [
            (circuit_one_qubit_one_param, np.array([0.1], is_input=True), {0.1: [-1.0, 0.0, 1.0]}),
            (
                circuit_one_qubit_one_param,
                np.array([-0.6], is_input=True),
                {-0.6: [-1.0, 0.0, 1.0]},
            ),
            (
                circuit_one_qubit_two_params,
                np.array([0.6, 0.3], is_input=True),
                {0.6: [-1.0, 0.0, 1.0], 0.3: [-1.0, 0.0, 1.0]},
            ),
            (
                circuit_two_qubits_two_params,
                np.array([0.6, 0.3], is_input=True),
                {0.6: [-1.0, 0.0, 1.0], 0.3: [-1.0, 0.0, 1.0]},
            ),
            (
                circuit_two_qubits_repeated_param,
                np.array([0.2], is_input=True),
                {0.2: [-2.0, -1.0, 0.0, 1.0, 2.0]},
            ),
            (
                circuit_two_qubits_controlled_rot,
                np.array([-1.0], is_input=True),
                {-1.0: [-2.0, -1.0, 0.0, 1.0, 2.0]},
            ),
        ],
    )
    def test_compute_fourier_spectra(self, circuit, inpt, spectra):
        """Test that Fourier spectra are correctly computed."""
        circuit(inpt)
        assert frequency_spectra(circuit.qtape) == spectra
