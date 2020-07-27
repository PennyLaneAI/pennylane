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
Unit tests for the :mod:`pennylane.qaoa` submodule.
"""
import pytest
import pennylane as qml
from pennylane import qaoa
from pennylane.templates import ApproxTimeEvolution


class TestUtils:
    """Tests that the utility functions are working properly"""

    @pytest.mark.parametrize(
        ("hamiltonian", "value"),
        (
            (qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)]), True),
            (qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]), False),
            (qml.Hamiltonian([1, 1], [qml.PauliZ(0) @ qml.Identity(1), qml.PauliZ(1)]), True),
            (qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1)]), False),
        ),
    )
    def test_diagonal_terms(self, hamiltonian, value):
        assert qaoa.layers._diagonal_terms(hamiltonian) == value


class TestLayers:
    """Tests that the cost and mixer layers are being constructed properly"""

    def test_mixer_layer_errors(self):
        """Tests that the mixer layer is throwing the correct errors"""

        hamiltonian = [[1, 1], [1, 1]]

        with pytest.raises(ValueError) as info:
            output = qaoa.mixer_layer(hamiltonian, wires=range(1))

        assert "hamiltonian must be of type pennylane.Hamiltonian, got list" in str(info.value)

    def test_cost_layer_errors(self):
        """Tests that the cost layer is throwing the correct errors"""

        hamiltonian = [[1, 1], [1, 1]]

        with pytest.raises(ValueError) as info:
            output = qaoa.cost_layer(hamiltonian, wires=range(1))

        assert "hamiltonian must be of type pennylane.Hamiltonian, got list" in str(info.value)

        hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(1)])

        with pytest.raises(ValueError) as info:
            output = qaoa.cost_layer(hamiltonian, wires=range(2))

        assert "hamiltonian must be written only in terms of PauliZ and Identity gates" in str(
            info.value
        )

    def test_mixer_layer_output(self):
        """Tests that the gates of the mixer layer is correct"""

        alpha = 1
        hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)])
        mixer = qaoa.mixer_layer(hamiltonian)

        with qml.utils.OperationRecorder() as rec1:
            mixer(alpha, wires=range(2))

        with qml.utils.OperationRecorder() as rec2:
            ApproxTimeEvolution(hamiltonian, 1, 1, range(2))

        for i, j in zip(rec1.operations, rec2.operations):

            prep = [i.name, i.parameters, i.wires]
            target = [j.name, j.parameters, j.wires]

            assert prep == target

    def test_cost_layer_output(self):
        """Tests that the gates of the cost layer is correct"""

    gamma = 1
    hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])
    cost = qaoa.cost_layer(hamiltonian)

    with qml.utils.OperationRecorder() as rec1:
        cost(gamma, wires=range(2))

    with qml.utils.OperationRecorder() as rec2:
        ApproxTimeEvolution(hamiltonian, 1, 1, range(2))

    for i, j in zip(rec1.operations, rec2.operations):
        prep = [i.name, i.parameters, i.wires]
        target = [j.name, j.parameters, j.wires]

        assert prep == target
