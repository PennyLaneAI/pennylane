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
import numpy as np
import pennylane as qml
from pennylane import qaoa
from networkx import Graph

graph = Graph()
graph.add_nodes_from([0, 1, 2])
graph.add_edges_from([(0, 1), (1, 2)])

#####################################################


class TestCostHamiltonians:
    """Tests that the cost Hamiltonians are being generated correctly"""

    def test_maxcut_error(self):
        """Tests that the MaxCut Hamiltonian throws the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(ValueError) as info:
            output = qaoa.MaxCut(graph)

        assert "Input graph must be a nx.Graph object, got list" in str(info.value)

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                Graph([(0, 1), (1, 2)]),
                qml.Hamiltonian(
                    [0.5, -0.5, 0.5, -0.5],
                    [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                    ],
                ),
            ),
            (
                Graph((np.array([0, 1]), np.array([1, 2]), np.array([0, 2]))),
                qml.Hamiltonian(
                    [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
                    [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(0) @ qml.Identity(2),
                        qml.PauliZ(0) @ qml.PauliZ(2),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                    ],
                ),
            ),
            (
                graph,
                qml.Hamiltonian(
                    [0.5, -0.5, 0.5, -0.5],
                    [
                        qml.Identity(0) @ qml.Identity(1),
                        qml.PauliZ(0) @ qml.PauliZ(1),
                        qml.Identity(1) @ qml.Identity(2),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                    ],
                ),
            ),
        ],
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
            cost_coeffs == target_coeffs and cost_ops == target_ops and cost_wires == target_wires
        )
