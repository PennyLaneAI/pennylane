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
import numpy as np
import pennylane as qml
from pennylane import qaoa


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
            mixer_coeffs == [1, 1, 1, 1]
            and mixer_ops == ["PauliX", "PauliX", "PauliX", "PauliX"]
            and mixer_wires == [0, 1, 2, 3]
        )

    def test_xy_mixer_type_error(self):
        """Tests that the XY mixer throws the correct error"""

        graph = 12

        with pytest.raises(ValueError) as info:
            output = qaoa.xy_mixer(graph)

        assert "Input graph must be a networkx.Graph object or Iterable, got int" in str(
            info.value
        )

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                [(0, 1), (1, 2)],
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(1),
                        qml.PauliY(0) @ qml.PauliY(1),
                        qml.PauliX(1) @ qml.PauliX(2),
                        qml.PauliY(1) @ qml.PauliY(2),
                    ],
                ),
            ),
            (
                (np.array([0, 1]), np.array([1, 2])),
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(1),
                        qml.PauliY(0) @ qml.PauliY(1),
                        qml.PauliX(1) @ qml.PauliX(2),
                        qml.PauliY(1) @ qml.PauliY(2),
                    ],
                ),
            ),
            (
                graph,
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(1),
                        qml.PauliY(0) @ qml.PauliY(1),
                        qml.PauliX(1) @ qml.PauliX(2),
                        qml.PauliY(1) @ qml.PauliY(2),
                    ],
                ),
            ),
        ],
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
            mixer_coeffs == target_coeffs
            and mixer_ops == target_ops
            and mixer_wires == target_wires
        )


class TestUtils:
    """Tests the QAOA utility functions"""

    @pytest.mark.parametrize(
        ("graph", "error"),
        [
            ([1, 2], "Elements of `graph` must be Iterable objects, got int"),
            (
                [(0, 1, 2), (2, 3)],
                "Elements of `graph` must be Iterable objects of length 2, got length 3",
            ),
            ([(0, 1), (1, 1)], "Edges must end in distinct nodes, got (1, 1)"),
            ([(0, 1), (1, 2), (1, 2)], "Nodes cannot be connected by more than one edge"),
        ],
    )
    def test_iterable_graph_errors(self, graph, error):
        """Tests that the `check_iterable_graph` method throws the correct errors"""

        with pytest.raises(ValueError) as info:
            output = qaoa.check_iterable_graph(graph)
        assert error in str(info.value)
