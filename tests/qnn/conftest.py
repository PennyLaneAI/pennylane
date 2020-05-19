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
        "w7": 0,
    }

    @qml.qnode(dev, interface=interface)
    def circuit(inputs, w1, w2, w3, w4, w5, w6, w7):
        """A circuit that embeds data using the AngleEmbedding and then performs a variety of
        operations. The output is a PauliZ measurement on the first output_dim qubits. One set of
        parameters, w5, are specified as non-trainable."""
        qml.templates.AngleEmbedding(inputs, wires=list(range(n_qubits)))
        qml.templates.StronglyEntanglingLayers(w1, wires=list(range(n_qubits)))
        qml.RX(w2[0], wires=0 % n_qubits)
        qml.RX(w3, wires=1 % n_qubits)
        qml.Rot(*w4, wires=2 % n_qubits)
        qml.templates.StronglyEntanglingLayers(w5, wires=list(range(n_qubits)))
        qml.Rot(*w6, wires=3 % n_qubits)
        qml.RX(w7, wires=4 % n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

    return circuit, weight_shapes
