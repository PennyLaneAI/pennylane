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
from pennylane import qaoa

#####################################################

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
                mixer_coeffs == [1 for i in wires] and
                mixer_ops == ['PauliX' for i in wires] and
                mixer_wires == list(wires)
        )

    def test_xy_mixer_error(self):
        """Tests that the XY mixer throws the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(ValueError) as info:
            output = qaoa.xy_mixer(graph)

        assert ("Inputted graph must be a `networkx.Graph` object, got list" in str(info.value))

    def test_xy_mixer_output(self):
        """Tests that the output of the XY mixer is correct"""

        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

        mixer_hamiltonian = qaoa.xy_mixer(graph)

        mixer_coeffs = mixer_hamiltonian.coeffs
        mixer_ops = [i.name for i in mixer_hamiltonian.ops]
        mixer_wires = [i.wires for i in mixer_hamiltonian.ops]

        gates = [['PauliX', 'PauliX'], ['PauliY', 'PauliY']]

        assert (
            mixer_coeffs == 2 * [0.5 for i in graph.nodes] and
            mixer_ops == [j for i in graph.edges for j in gates] and
            mixer_wires == [list(i) for i in graph.edges for j in range(2)]
        )

class TestCostHamiltonians:
    """Tests that the cost Hamiltonians are being generated correctly"""

    def test_maxcut_error(self):

    def test_maxcut_output(self):

class TestLayers:
    """Tests that the QAOA layers are being generated correctly"""