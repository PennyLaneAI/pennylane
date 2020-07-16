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
import networkx as nx
import pennylane as qml
from pennylane import qaoa
import numpy as np

#####################################################

graph = nx.Graph()
graph.add_nodes_from([0, 1, 2])
graph.add_edges_from([(0, 1), (1, 2)])

class TestMixerHamiltonians:
    """Tests that the mixer Hamiltonians are being generated correctly"""

    def test_x_mixer_output(self):
        """Tests that the output of the Pauli-X mixer is correct"""

        num_qubits = 4
        wires = range(num_qubits)

        mixer_hamiltonian = qaoa.x_mixer(wires)

        mixer_coeffs = mixer_hamiltonian.coeffs
        mixer_ops = [i.name for i in mixer_hamiltonian.ops]
        mixer_wires = [i.wires[0] for i in mixer_hamiltonian.ops]

        assert (
                mixer_coeffs == [1, 1, 1, 1] and
                mixer_ops == ['PauliX', 'PauliX', 'PauliX', 'PauliX'] and
                mixer_wires == [0, 1, 2, 3]
        )

    def test_xy_mixer_type_error(self):
        """Tests that the XY mixer throws the correct error"""

        graph = 12

        with pytest.raises(ValueError) as info:
            output = qaoa.xy_mixer(graph)

        assert ("Inputted graph must be a networkx.Graph object or Iterable, got int" in str(info.value))

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                [(0, 1), (1, 2)], qml.Hamiltonian([0.5, 0.5, 0.5, 0.5], [
                    qml.PauliX(0) @ qml.PauliX(1),
                    qml.PauliY(0) @ qml.PauliY(1),
                    qml.PauliX(1) @ qml.PauliX(2),
                    qml.PauliY(1) @ qml.PauliY(2)
            ])
             ),
            (
                (np.array([0, 1]), np.array([1, 2])), qml.Hamiltonian([0.5, 0.5, 0.5, 0.5], [
                    qml.PauliX(0) @ qml.PauliX(1),
                    qml.PauliY(0) @ qml.PauliY(1),
                    qml.PauliX(1) @ qml.PauliX(2),
                    qml.PauliY(1) @ qml.PauliY(2)
                ])
            ),
            (
                graph, qml.Hamiltonian([0.5, 0.5, 0.5, 0.5], [
                    qml.PauliX(0) @ qml.PauliX(1),
                    qml.PauliY(0) @ qml.PauliY(1),
                    qml.PauliX(1) @ qml.PauliX(2),
                    qml.PauliY(1) @ qml.PauliY(2)
                ])
            )
        ]
    )
    def test_xy_mixer_output(self, graph, target_hamiltonian):
        """Tests that the output of the XY mixer is correct"""

        mixer_hamiltonian = qaoa.xy_mixer(graph)

        mixer_coeffs = mixer_hamiltonian.coeffs
        mixer_ops = [i.name for i in mixer_hamiltonian.ops]
        mixer_wires = [i.wires for i in mixer_hamiltonian.ops]

        target_coeffs = target_hamiltonian.coeffs
        target_ops = [i.name for i in target_hamiltonian.ops]
        target_wires = [i.wires for i in target_hamiltonian.ops]

        assert (
            mixer_coeffs == target_coeffs and
            mixer_ops == target_ops and
            mixer_wires == target_wires
        )

class TestCostHamiltonians:
    """Tests that the cost Hamiltonians are being generated correctly"""

    def test_maxcut_error(self):
        """Tests that the MaxCut Hamiltonian throws the correct error"""

        graph = 12

        with pytest.raises(ValueError) as info:
            output = qaoa.MaxCut(graph)

        assert ("Inputted graph must be a networkx.Graph object or Iterable, got int" in str(info.value))

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                    [(0, 1), (1, 2)],
                    qml.Hamiltonian([0.5, -0.5, 0.5, -0.5], [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2)
                    ])
            ),
            (
                    (np.array([0, 1]), np.array([1, 2]), np.array([0, 2])),
                    qml.Hamiltonian([0.5, -0.5, 0.5, -0.5, 0.5, -0.5], [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                        qml.Identity(0) @ qml.Identity(2),
                        qml.PauliZ(0) @ qml.PauliZ(2)
                    ])
            ),
            (
                    graph,
                    qml.Hamiltonian([0.5, -0.5, 0.5, -0.5], [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2)
                    ])
            )
        ]
    )
    def test_maxcut_output(self, graph, target_hamiltonian):
        """Tests that the output of the MaxCut method is correct"""

        cost_hamiltonian = qaoa.MaxCut(graph)

        cost_coeffs = cost_hamiltonian.coeffs
        cost_ops = [i.name for i in cost_hamiltonian.ops]
        cost_wires = [i.wires for i in cost_hamiltonian.ops]

        target_coeffs = target_hamiltonian.coeffs
        target_ops = [i.name for i in target_hamiltonian.ops]
        target_wires = [i.wires for i in target_hamiltonian.ops]

        assert (
                cost_coeffs == target_coeffs and
                cost_ops == target_ops and
                cost_wires == target_wires
        )

class TestLayers:
    """Tests that the cost and mixer layers are being constructed properly"""

    def test_mixer_layer_errors(self):
        """Tests that the mixer layer is throwing the correct errors"""

        hamiltonian = [[1, 1], [1, 1]]

        with pytest.raises(ValueError) as info:
            output = qaoa.mixer_layer(hamiltonian)

        assert (
            "`hamiltonian` must be of type pennylane.Hamiltonian, got list"
        )

'''
    def test_cost_layer_errors(self):
        """Tests that the cost layer is throwing the correct errors"""
'''

'''
    def test_cost_layer_output(self):
        """Tests that the output of the cost layer is correct"""

    def test_mixer_layer_output(self):
        """Tests that the output of the mixer layer is correct"""
'''