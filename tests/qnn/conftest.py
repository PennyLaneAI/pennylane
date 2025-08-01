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
Common fixtures for the qnn module.
"""
import numpy as np
import pytest

import pennylane as qml


@pytest.fixture
def get_circuit(n_qubits, output_dim, interface):
    """Fixture for getting a sample quantum circuit with a controllable qubit number and output
    dimension. Returns both the circuit and the shape of the weights."""

    dev = qml.device("default.qubit", wires=n_qubits)
    weight_shapes = {
        "w1": (3, n_qubits, 3),
        "w2": (1,),
        "w3": 1,
        "w4": [3],
        "w5": (2, n_qubits, 3),
        "w6": 3,
        "w7": 1,
    }

    # pylint: disable=too-many-arguments
    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1, w2, w3, w4, w5, w6, w7):
        """A circuit that embeds data using the AngleEmbedding and then performs a variety of
        operations. The output is a PauliZ measurement on the first output_dim qubits. One set of
        parameters, w5, are specified as non-trainable."""
        qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
        qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
        qml.RX(w2[0], wires=0 % n_qubits)
        qml.RX(w3, wires=1 % n_qubits)
        qml.Rot(w4[0], w4[1], w4[2], wires=2 % n_qubits)
        qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
        qml.Rot(w6[0], w6[1], w6[2], wires=3 % n_qubits)
        qml.RX(w7, wires=4 % n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

    return circuit, weight_shapes


@pytest.fixture
def get_circuit_dm(n_qubits, output_dim, interface):
    """Fixture for getting a sample quantum circuit with a controllable qubit number and output
    dimension for density matrix return type. Returns both the circuit and the shape of the weights.
    """

    dev = qml.device("default.qubit", wires=n_qubits)
    weight_shapes = {
        "w1": (3, n_qubits, 3),
        "w2": (1,),
        "w3": 1,
        "w4": [3],
        "w5": (2, n_qubits, 3),
        "w6": 3,
        "w7": 1,
    }

    # pylint: disable=too-many-arguments
    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1, w2, w3, w4, w5, w6, w7):
        """Sample circuit to be used for testing density_matrix() return type."""
        qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
        qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
        qml.RX(w2[0], wires=0 % n_qubits)
        qml.RX(w3, wires=1 % n_qubits)
        qml.Rot(w4[0], w4[1], w4[2], wires=2 % n_qubits)
        qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
        qml.Rot(w6[0], w6[1], w6[2], wires=3 % n_qubits)
        qml.RX(w7, wires=4 % n_qubits)

        # Using np.log2() here because output_dim is sampled from varying the number of
        # qubits (say, nq) and calculated as (2 ** nq, 2 ** nq)
        return qml.density_matrix(wires=list(range(int(np.log2(output_dim[0])))))

    return circuit, weight_shapes


@pytest.fixture
def get_circuit_shots(n_qubits, output_dim, interface, shots):
    """Fixture for getting a circuit with shots specified."""

    dev = qml.device("default.qubit", wires=n_qubits)
    weight_shapes = {
        "w1": (3, n_qubits, 3),
        "w2": (1,),
        "w3": 1,
        "w4": [3],
        "w5": (2, n_qubits, 3),
        "w6": 3,
        "w7": 1,
    }

    # pylint: disable=too-many-arguments
    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1, w2, w3, w4, w5, w6, w7):
        """A circuit that embeds data using the AngleEmbedding and then performs a variety of
        operations. The output is a PauliZ measurement on the first output_dim qubits. One set of
        parameters, w5, are specified as non-trainable."""
        qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
        qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
        qml.RX(w2[0], wires=0 % n_qubits)
        qml.RX(w3, wires=1 % n_qubits)
        qml.Rot(w4[0], w4[1], w4[2], wires=2 % n_qubits)
        qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
        qml.Rot(w6[0], w6[1], w6[2], wires=3 % n_qubits)
        qml.RX(w7, wires=4 % n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

    return circuit, weight_shapes
