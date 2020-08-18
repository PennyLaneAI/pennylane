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
from pennylane.templates import ApproxTimeEvolution
from pennylane.wires import Wires


#####################################################

graph = Graph()
graph.add_nodes_from([0, 1, 2])
graph.add_edges_from([(0, 1), (1, 2)])

non_consecutive_graph = Graph([(0, 4), (3, 4), (2, 1), (2, 0)])


class TestMixerHamiltonians:
    """Tests that the mixer Hamiltonians are being generated correctly"""

    def test_x_mixer_output(self):
        """Tests that the output of the Pauli-X mixer is correct"""

        wires = range(4)
        mixer_hamiltonian = qaoa.x_mixer(wires)

        mixer_coeffs = mixer_hamiltonian.coeffs
        mixer_ops = [i.name for i in mixer_hamiltonian.ops]
        mixer_wires = [i.wires[0] for i in mixer_hamiltonian.ops]

        assert mixer_coeffs == [1, 1, 1, 1]
        assert mixer_ops == ["PauliX", "PauliX", "PauliX", "PauliX"]
        assert mixer_wires == [Wires(0), Wires(1), Wires(2), Wires(3)]

    def test_xy_mixer_type_error(self):
        """Tests that the XY mixer throws the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(ValueError, match=r"Input graph must be a nx.Graph object, got list"):
            qaoa.xy_mixer(graph)

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                Graph([(0, 1), (1, 2), (2, 3)]),
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(1),
                        qml.PauliY(0) @ qml.PauliY(1),
                        qml.PauliX(1) @ qml.PauliX(2),
                        qml.PauliY(1) @ qml.PauliY(2),
                        qml.PauliX(2) @ qml.PauliX(3),
                        qml.PauliY(2) @ qml.PauliY(3),
                    ],
                ),
            ),
            (
                Graph((np.array([0, 1]), np.array([1, 2]), np.array([2, 0]))),
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(1),
                        qml.PauliY(0) @ qml.PauliY(1),
                        qml.PauliX(0) @ qml.PauliX(2),
                        qml.PauliY(0) @ qml.PauliY(2),
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
            (
                non_consecutive_graph,
                qml.Hamiltonian(
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [
                        qml.PauliX(0) @ qml.PauliX(4),
                        qml.PauliY(0) @ qml.PauliY(4),
                        qml.PauliX(0) @ qml.PauliX(2),
                        qml.PauliY(0) @ qml.PauliY(2),
                        qml.PauliX(4) @ qml.PauliX(3),
                        qml.PauliY(4) @ qml.PauliY(3),
                        qml.PauliX(2) @ qml.PauliX(1),
                        qml.PauliY(2) @ qml.PauliY(1),
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

        assert mixer_coeffs == target_coeffs
        assert mixer_ops == target_ops
        assert mixer_wires == target_wires


GRAPHS = [
    Graph([(0, 1), (1, 2)]),
    Graph((np.array([0, 1]), np.array([1, 2]), np.array([0, 2]))),
    graph,
]

COST_COEFFS = [[-0.5, 0.5, -0.5, 0.5], [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], [-0.5, 0.5, -0.5, 0.5]]

COST_TERMS = [
    [
        qml.Identity(0) @ qml.Identity(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.Identity(1) @ qml.Identity(2),
        qml.PauliZ(1) @ qml.PauliZ(2),
    ],
    [
        qml.Identity(0) @ qml.Identity(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.Identity(0) @ qml.Identity(2),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.Identity(1) @ qml.Identity(2),
        qml.PauliZ(1) @ qml.PauliZ(2),
    ],
    [
        qml.Identity(0) @ qml.Identity(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.Identity(1) @ qml.Identity(2),
        qml.PauliZ(1) @ qml.PauliZ(2),
    ],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(3)]

MIXER_COEFFS = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

MIXER_TERMS = [
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(3)]

MAXCUT = zip(GRAPHS, COST_HAMILTONIANS, MIXER_HAMILTONIANS)

def decompose_hamiltonian(hamiltonian):

    coeffs = hamiltonian.coeffs
    ops = [i.name for i in hamiltonian.ops]
    wires = [i.wires for i in hamiltonian.ops]

    return [coeffs, ops, wires]


class TestCostHamiltonians:
    """Tests that the cost Hamiltonians are being generated correctly"""

    def test_maxcut_error(self):
        """Tests that the MaxCut Hamiltonian throws the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(ValueError, match=r"nput graph must be a nx\.Graph"):
            qaoa.maxcut(graph)

    @pytest.mark.parametrize(("graph", "cost_hamiltonian", "mixer_hamiltonian"), MAXCUT)
    def test_maxcut_output(self, graph, cost_hamiltonian, mixer_hamiltonian):
        """Tests that the output of the MaxCut method is correct"""

        cost_h, mixer_h = qaoa.maxcut(graph)

        assert decompose_hamiltonian(cost_hamiltonian) == decompose_hamiltonian(cost_h)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)


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

        with pytest.raises(ValueError, match=r"hamiltonian must be of type pennylane.Hamiltonian"):
            qaoa.mixer_layer(0.1, hamiltonian)

    def test_cost_layer_errors(self):
        """Tests that the cost layer is throwing the correct errors"""

        hamiltonian = [[1, 1], [1, 1]]

        with pytest.raises(ValueError, match=r"hamiltonian must be of type pennylane.Hamiltonian"):
            qaoa.cost_layer(0.1, hamiltonian)

        hamiltonian = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(1)])

        with pytest.raises(ValueError, match=r"hamiltonian must be written only in terms of PauliZ and Identity gates"):
            qaoa.cost_layer(0.1, hamiltonian)

    @pytest.mark.parametrize(
        ("mixer", "gates"),
        [
            [
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                [qml.PauliRot(2, "X", wires=[0]), qml.PauliRot(2, "X", wires=[1])]
            ],
            [
                qaoa.xy_mixer(Graph([(0, 1), (1, 2), (2, 0)])),
                [qml.PauliRot(1, "XX", wires=[0, 1]), qml.PauliRot(1, "YY", wires=[0, 1]),
                 qml.PauliRot(1, "XX", wires=[0, 2]), qml.PauliRot(1, "YY", wires=[0, 2]),
                 qml.PauliRot(1, "XX", wires=[1, 2]), qml.PauliRot(1, "YY", wires=[1, 2])]
            ]
        ]
    )
    def test_mixer_layer_output(self, mixer, gates):
        """Tests that the gates of the mixer layer are correct"""

        alpha = 1

        with qml._queuing.OperationRecorder() as rec:
            qaoa.mixer_layer(alpha, mixer)

        for i, j in zip(rec.operations, gates):

            prep = [i.name, i.parameters, i.wires]
            target = [j.name, j.parameters, j.wires]

            assert prep == target

    @pytest.mark.parametrize(
        ("cost", "gates"),
        [
            [
                qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)]),
                [qml.PauliRot(2, "Z", wires=[0]), qml.PauliRot(2, "Z", wires=[1])]
            ],
            [
                qaoa.maxcut(Graph([(0, 1), (1, 2), (2, 0)]))[0],
                [qml.PauliRot(1, "ZZ", wires=[0, 1]),
                 qml.PauliRot(1, "ZZ", wires=[0, 2]),
                 qml.PauliRot(1, "ZZ", wires=[1, 2])]
            ]
        ]
    )
    def test_cost_layer_output(self, cost, gates):
        """Tests that the gates of the cost layer is correct"""

        gamma = 1

        with qml._queuing.OperationRecorder() as rec:
            qaoa.cost_layer(gamma, cost)

        for i, j in zip(rec.operations, gates):
            prep = [i.name, i.parameters, i.wires]
            target = [j.name, j.parameters, j.wires]

        assert prep == target
