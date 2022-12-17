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
import itertools
import numpy as np

import networkx as nx
from networkx import Graph
import retworkx as rx

import pennylane as qml
from pennylane import qaoa

from pennylane.qaoa.cycle import (
    edges_to_wires,
    wires_to_edges,
    _inner_net_flow_constraint_hamiltonian,
    net_flow_constraint,
    loss_hamiltonian,
    _square_hamiltonian_terms,
    cycle_mixer,
    _partial_cycle_mixer,
    out_flow_constraint,
    _inner_out_flow_constraint_hamiltonian,
)
from scipy.linalg import expm
from scipy.sparse import csc_matrix, kron


#####################################################

graph = Graph()
graph.add_nodes_from([0, 1, 2])
graph.add_edges_from([(0, 1), (1, 2)])

graph_rx = rx.PyGraph()
graph_rx.add_nodes_from([0, 1, 2])
graph_rx.add_edges_from([(0, 1, ""), (1, 2, "")])

non_consecutive_graph = Graph([(0, 4), (3, 4), (2, 1), (2, 0)])
non_consecutive_graph_rx = rx.PyGraph()
non_consecutive_graph_rx.add_nodes_from([0, 1, 2, 3, 4])
non_consecutive_graph_rx.add_edges_from([(0, 4, ""), (0, 2, ""), (4, 3, ""), (2, 1, "")])

g1 = Graph([(0, 1), (1, 2)])

g1_rx = rx.PyGraph()
g1_rx.add_nodes_from([0, 1, 2])
g1_rx.add_edges_from([(0, 1, ""), (1, 2, "")])

g2 = nx.Graph([(0, 1), (1, 2), (2, 3)])

g2_rx = rx.PyGraph()
g2_rx.add_nodes_from([0, 1, 2, 3])
g2_rx.add_edges_from([(0, 1, ""), (1, 2, ""), (2, 3, "")])

b_rx = rx.PyGraph()
b_rx.add_nodes_from(["b", 1, 0.3])
b_rx.add_edges_from([(0, 1, ""), (1, 2, ""), (0, 2, "")])


def catch_warn_ExpvalCost(ansatz, hamiltonian, device, **kwargs):
    """Computes the ExpvalCost and catches the initial deprecation warning."""

    with pytest.warns(UserWarning, match="is deprecated,"):
        res = qml.ExpvalCost(ansatz, hamiltonian, device, **kwargs)
    return res


def decompose_hamiltonian(hamiltonian):
    coeffs = list(qml.math.toarray(hamiltonian.coeffs))
    ops = [i.name for i in hamiltonian.ops]
    wires = [i.wires for i in hamiltonian.ops]
    return [coeffs, ops, wires]


def lollipop_graph_rx(mesh_nodes: int, path_nodes: int, to_directed: bool = False):
    if to_directed:
        g = rx.generators.directed_mesh_graph(weights=[*range(mesh_nodes)])
    else:
        g = rx.generators.mesh_graph(weights=[*range(mesh_nodes)])
    if path_nodes < 1:
        return g

    for i in range(path_nodes):
        g.add_node(mesh_nodes + i)
        g.add_edges_from([(mesh_nodes + i - 1, mesh_nodes + i, "")])
        if to_directed:
            g.add_edges_from([(mesh_nodes + i, mesh_nodes + i - 1, "")])
    return g


def matrix(hamiltonian: qml.Hamiltonian, n_wires: int) -> csc_matrix:
    r"""Calculates the matrix representation of an input Hamiltonian in the standard basis.

    Args:
        hamiltonian (qml.Hamiltonian): the input Hamiltonian
        n_wires (int): the total number of wires

    Returns:
        csc_matrix: a sparse matrix representation
    """
    ops_matrices = []

    for op in hamiltonian.ops:
        op_wires = np.array(op.wires.tolist())
        op_list = op.non_identity_obs if isinstance(op, qml.operation.Tensor) else [op]
        op_matrices = []

        for wire in range(n_wires):
            loc = np.argwhere(op_wires == wire).flatten()
            mat = np.eye(2) if len(loc) == 0 else op_list[loc[0]].matrix()
            mat = csc_matrix(mat)
            op_matrices.append(mat)

        op_matrix = op_matrices.pop(0)

        for mat in op_matrices:
            op_matrix = kron(op_matrix, mat)

        ops_matrices.append(op_matrix)

    mat = sum(coeff * op_mat for coeff, op_mat in zip(hamiltonian.coeffs, ops_matrices))
    return csc_matrix(mat)


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
        assert mixer_wires == [0, 1, 2, 3]

    @pytest.mark.parametrize("constrained", [True, False])
    def test_x_mixer_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        wires = range(4)
        mixer_hamiltonian = qaoa.x_mixer(wires)

        # check that all observables commute
        assert all(qml.is_commuting(o, mixer_hamiltonian.ops[0]) for o in mixer_hamiltonian.ops[1:])
        # check that the 1-group grouping information was set
        assert mixer_hamiltonian.grouping_indices is not None
        assert mixer_hamiltonian.grouping_indices == [[0, 1, 2, 3]]

    def test_xy_mixer_type_error(self):
        """Tests that the XY mixer throws the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph object, got list"
        ):
            qaoa.xy_mixer(graph)

    @pytest.mark.parametrize(
        ("graph", "target_hamiltonian"),
        [
            (
                g2,
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
            (
                g2_rx,
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
                graph_rx,
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
                non_consecutive_graph_rx,
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

    def test_bit_flip_mixer_errors(self):
        """Tests that the bit-flip mixer throws the correct errors"""

        graph = [(0, 1), (1, 2)]
        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph object"
        ):
            qaoa.bit_flip_mixer(graph, 0)

        n = 2
        with pytest.raises(ValueError, match=r"'b' must be either 0 or 1"):
            qaoa.bit_flip_mixer(Graph(graph), n)

    @pytest.mark.parametrize(
        ("graph", "n", "target_hamiltonian"),
        [
            (
                Graph([(0, 1)]),
                1,
                qml.Hamiltonian(
                    [0.5, -0.5, 0.5, -0.5],
                    [
                        qml.PauliX(0),
                        qml.PauliX(0) @ qml.PauliZ(1),
                        qml.PauliX(1),
                        qml.PauliX(1) @ qml.PauliZ(0),
                    ],
                ),
            ),
            (
                g1,
                0,
                qml.Hamiltonian(
                    [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    [
                        qml.PauliX(0),
                        qml.PauliX(0) @ qml.PauliZ(1),
                        qml.PauliX(1),
                        qml.PauliX(1) @ qml.PauliZ(2),
                        qml.PauliX(1) @ qml.PauliZ(0),
                        qml.PauliX(1) @ qml.PauliZ(0) @ qml.PauliZ(2),
                        qml.PauliX(2),
                        qml.PauliX(2) @ qml.PauliZ(1),
                    ],
                ),
            ),
            (
                g1_rx,
                0,
                qml.Hamiltonian(
                    [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    [
                        qml.PauliX(0),
                        qml.PauliX(0) @ qml.PauliZ(1),
                        qml.PauliX(1),
                        qml.PauliX(1) @ qml.PauliZ(2),
                        qml.PauliX(1) @ qml.PauliZ(0),
                        qml.PauliX(1) @ qml.PauliZ(0) @ qml.PauliZ(2),
                        qml.PauliX(2),
                        qml.PauliX(2) @ qml.PauliZ(1),
                    ],
                ),
            ),
            (
                Graph([("b", 1), (1, 0.3), (0.3, "b")]),
                1,
                qml.Hamiltonian(
                    [0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25],
                    [
                        qml.PauliX("b"),
                        qml.PauliX("b") @ qml.PauliZ(0.3),
                        qml.PauliX("b") @ qml.PauliZ(1),
                        qml.PauliX("b") @ qml.PauliZ(1) @ qml.PauliZ(0.3),
                        qml.PauliX(1),
                        qml.PauliX(1) @ qml.PauliZ(0.3),
                        qml.PauliX(1) @ qml.PauliZ("b"),
                        qml.PauliX(1) @ qml.PauliZ("b") @ qml.PauliZ(0.3),
                        qml.PauliX(0.3),
                        qml.PauliX(0.3) @ qml.PauliZ("b"),
                        qml.PauliX(0.3) @ qml.PauliZ(1),
                        qml.PauliX(0.3) @ qml.PauliZ(1) @ qml.PauliZ("b"),
                    ],
                ),
            ),
            (
                b_rx,
                1,
                qml.Hamiltonian(
                    [0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25],
                    [
                        qml.PauliX("b"),
                        qml.PauliX("b") @ qml.PauliZ(0.3),
                        qml.PauliX("b") @ qml.PauliZ(1),
                        qml.PauliX("b") @ qml.PauliZ(1) @ qml.PauliZ(0.3),
                        qml.PauliX(1),
                        qml.PauliX(1) @ qml.PauliZ(0.3),
                        qml.PauliX(1) @ qml.PauliZ("b"),
                        qml.PauliX(1) @ qml.PauliZ("b") @ qml.PauliZ(0.3),
                        qml.PauliX(0.3),
                        qml.PauliX(0.3) @ qml.PauliZ(1),
                        qml.PauliX(0.3) @ qml.PauliZ("b"),
                        qml.PauliX(0.3) @ qml.PauliZ("b") @ qml.PauliZ(1),
                    ],
                ),
            ),
        ],
    )
    def test_bit_flip_mixer_output(self, graph, n, target_hamiltonian):
        """Tests that the output of the bit-flip mixer is correct"""

        mixer_hamiltonian = qaoa.bit_flip_mixer(graph, n)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(target_hamiltonian)


"""GENERATES CASES TO TEST THE MAXCUT PROBLEM"""

GRAPHS = [
    g1,
    g1_rx,
    Graph((np.array([0, 1]), np.array([1, 2]), np.array([0, 2]))),
    graph,
    graph_rx,
]

COST_COEFFS = [
    [0.5, 0.5, -1.0],
    [0.5, 0.5, -1.0],
    [0.5, 0.5, 0.5, -1.5],
    [0.5, 0.5, -1.0],
    [0.5, 0.5, -1.0],
]

COST_TERMS = [
    [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2), qml.Identity(0)],
    [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2), qml.Identity(0)],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.Identity(0),
    ],
    [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2), qml.Identity(0)],
    [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(1) @ qml.PauliZ(2), qml.Identity(0)],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(5)]

MIXER_COEFFS = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]

MIXER_TERMS = [
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(5)]

MAXCUT = list(zip(GRAPHS, COST_HAMILTONIANS, MIXER_HAMILTONIANS))

"""GENERATES THE CASES TO TEST THE MAX INDEPENDENT SET PROBLEM"""

CONSTRAINED = [
    True,
    True,
    True,
    False,
    False,
]

COST_COEFFS = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0.75, 0.25, -0.5, 0.75, 0.25],
    [0.75, 0.25, -0.5, 0.75, 0.25],
]

COST_TERMS = [
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2),
    ],
    # [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2),
    ],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(5)]

MIXER_COEFFS = [
    [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
    [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    [1, 1, 1],
    [1, 1, 1],
]

MIXER_TERMS = [
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliX(1),
        qml.PauliX(1) @ qml.PauliZ(2),
        qml.PauliX(1) @ qml.PauliZ(0),
        qml.PauliX(1) @ qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliZ(1),
    ],
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliX(1),
        qml.PauliX(1) @ qml.PauliZ(2),
        qml.PauliX(1) @ qml.PauliZ(0),
        qml.PauliX(1) @ qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliZ(1),
    ],
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliZ(2),
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliX(1),
        qml.PauliX(1) @ qml.PauliZ(2),
        qml.PauliX(1) @ qml.PauliZ(0),
        qml.PauliX(1) @ qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliZ(0),
        qml.PauliX(2) @ qml.PauliZ(1),
        qml.PauliX(2) @ qml.PauliZ(1) @ qml.PauliZ(0),
    ],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(5)]

MIS = list(zip(GRAPHS, CONSTRAINED, COST_HAMILTONIANS, MIXER_HAMILTONIANS))

"""GENERATES THE CASES TO TEST THE MIN VERTEX COVER PROBLEM"""

COST_COEFFS = [
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [0.75, -0.25, 0.5, 0.75, -0.25],
    [0.75, -0.25, 0.5, 0.75, -0.25],
]

COST_TERMS = [
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2),
    ],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2),
    ],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(5)]

MIXER_COEFFS = [
    [0.5, -0.5, 0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
    [0.5, -0.5, 0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
    [0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25],
    [1, 1, 1],
    [1, 1, 1],
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(5)]

MVC = list(zip(GRAPHS, CONSTRAINED, COST_HAMILTONIANS, MIXER_HAMILTONIANS))

"""GENERATES THE CASES TO TEST THE MAXCLIQUE PROBLEM"""

COST_COEFFS = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0.75, 0.25, 0.25, 1],
    [0.75, 0.25, 0.25, 1],
]

COST_TERMS = [
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)],
    [qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(0), qml.PauliZ(2), qml.PauliZ(1)],
    [qml.PauliZ(0) @ qml.PauliZ(2), qml.PauliZ(0), qml.PauliZ(2), qml.PauliZ(1)],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(5)]

MIXER_COEFFS = [
    [0.5, 0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [1, 1, 1],
    [1, 1, 1],
]

MIXER_TERMS = [
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliZ(2),
        qml.PauliX(1),
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliZ(0),
    ],
    [
        qml.PauliX(0),
        qml.PauliX(0) @ qml.PauliZ(2),
        qml.PauliX(1),
        qml.PauliX(2),
        qml.PauliX(2) @ qml.PauliZ(0),
    ],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
    [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)],
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(5)]

MAXCLIQUE = list(zip(GRAPHS, CONSTRAINED, COST_HAMILTONIANS, MIXER_HAMILTONIANS))

"""GENERATES CASES TO TEST EDGE DRIVER COST HAMILTONIAN"""
GRAPHS = GRAPHS[1:-2]
GRAPHS.append(graph)
GRAPHS.append(Graph([("b", 1), (1, 2.3)]))
GRAPHS.append(graph_rx)

b1_rx = rx.PyGraph()
b1_rx.add_nodes_from(["b", 1, 2.3])
b1_rx.add_edges_from([(0, 1, ""), (1, 2, "")])

GRAPHS.append(b1_rx)

REWARDS = [
    ["00"],
    ["00", "11"],
    ["00", "11", "01", "10"],
    ["00", "01", "10"],
    ["00", "11", "01", "10"],
    ["00", "01", "10"],
]

HAMILTONIANS = [
    qml.Hamiltonian(
        [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25],
        [
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(1) @ qml.PauliZ(2),
            qml.PauliZ(1),
            qml.PauliZ(2),
        ],
    ),
    qml.Hamiltonian(
        [-0.5, -0.5, -0.5],
        [
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(2),
            qml.PauliZ(1) @ qml.PauliZ(2),
        ],
    ),
    qml.Hamiltonian([1, 1, 1], [qml.Identity(0), qml.Identity(1), qml.Identity(2)]),
    qml.Hamiltonian(
        [0.25, -0.25, -0.25, 0.25, -0.25, -0.25],
        [
            qml.PauliZ("b") @ qml.PauliZ(1),
            qml.PauliZ("b"),
            qml.PauliZ(1),
            qml.PauliZ(1) @ qml.PauliZ(2.3),
            qml.PauliZ(1),
            qml.PauliZ(2.3),
        ],
    ),
    qml.Hamiltonian([1, 1, 1], [qml.Identity(0), qml.Identity(1), qml.Identity(2)]),
    qml.Hamiltonian(
        [0.25, -0.25, -0.25, 0.25, -0.25, -0.25],
        [
            qml.PauliZ("b") @ qml.PauliZ(1),
            qml.PauliZ("b"),
            qml.PauliZ(1),
            qml.PauliZ(1) @ qml.PauliZ(2.3),
            qml.PauliZ(1),
            qml.PauliZ(2.3),
        ],
    ),
]

EDGE_DRIVER = zip(GRAPHS, REWARDS, HAMILTONIANS)

"""GENERATES THE CASES TO TEST THE MAXIMUM WEIGHTED CYCLE PROBLEM"""
digraph_complete = nx.complete_graph(3).to_directed()
complete_edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(digraph_complete.edges)}
for k, v in complete_edge_weight_data.items():
    digraph_complete[k[0]][k[1]]["weight"] = v

digraph_complete_rx = rx.generators.directed_mesh_graph(3, [0, 1, 2])
complete_edge_weight_data = {
    edge: (i + 1) * 0.5 for i, edge in enumerate(sorted(digraph_complete_rx.edge_list()))
}
for k, v in complete_edge_weight_data.items():
    digraph_complete_rx.update_edge(k[0], k[1], {"weight": v})

DIGRAPHS = [digraph_complete] * 2

MWC_CONSTRAINED = [True, False]

COST_COEFFS = [
    [
        -0.6931471805599453,
        0.0,
        0.4054651081081644,
        0.6931471805599453,
        0.9162907318741551,
        1.0986122886681098,
    ],
    [
        -6.693147180559945,
        -6.0,
        -5.594534891891835,
        -5.306852819440055,
        -5.083709268125845,
        -4.90138771133189,
        54,
        12,
        -12,
        -6,
        -6,
        -12,
        6,
        12,
        -6,
        -6,
        -12,
        6,
        12,
        -6,
        -6,
        6,
    ],
]

COST_TERMS = [
    [
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[3]),
        qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[5]),
    ],
    [
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[3]),
        qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[5]),
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
        qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[5]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
        qml.PauliZ(wires=[3]) @ qml.PauliZ(wires=[5]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[5]),
        qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[5]),
        qml.PauliZ(wires=[3]) @ qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[5]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
    ],
]

COST_HAMILTONIANS = [qml.Hamiltonian(COST_COEFFS[i], COST_TERMS[i]) for i in range(2)]

MIXER_COEFFS = [
    [
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
        0.25,
        0.25,
        0.25,
        -0.25,
    ],
    [1] * 6,
]

MIXER_TERMS = [
    [
        qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[5]),
        qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[5]),
        qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[1]) @ qml.PauliX(wires=[0]) @ qml.PauliX(wires=[3]),
        qml.PauliY(wires=[1]) @ qml.PauliY(wires=[0]) @ qml.PauliX(wires=[3]),
        qml.PauliY(wires=[1]) @ qml.PauliX(wires=[0]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliY(wires=[0]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[2]) @ qml.PauliX(wires=[3]) @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[2]) @ qml.PauliY(wires=[3]) @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[2]) @ qml.PauliX(wires=[3]) @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[2]) @ qml.PauliY(wires=[3]) @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[3]) @ qml.PauliX(wires=[2]) @ qml.PauliX(wires=[1]),
        qml.PauliY(wires=[3]) @ qml.PauliY(wires=[2]) @ qml.PauliX(wires=[1]),
        qml.PauliY(wires=[3]) @ qml.PauliX(wires=[2]) @ qml.PauliY(wires=[1]),
        qml.PauliX(wires=[3]) @ qml.PauliY(wires=[2]) @ qml.PauliY(wires=[1]),
        qml.PauliX(wires=[4]) @ qml.PauliX(wires=[5]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[4]) @ qml.PauliY(wires=[5]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[4]) @ qml.PauliX(wires=[5]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[4]) @ qml.PauliY(wires=[5]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[5]) @ qml.PauliX(wires=[4]) @ qml.PauliX(wires=[0]),
        qml.PauliY(wires=[5]) @ qml.PauliY(wires=[4]) @ qml.PauliX(wires=[0]),
        qml.PauliY(wires=[5]) @ qml.PauliX(wires=[4]) @ qml.PauliY(wires=[0]),
        qml.PauliX(wires=[5]) @ qml.PauliY(wires=[4]) @ qml.PauliY(wires=[0]),
    ],
    [qml.PauliX(wires=i) for i in range(6)],
]

MIXER_HAMILTONIANS = [qml.Hamiltonian(MIXER_COEFFS[i], MIXER_TERMS[i]) for i in range(2)]

MAPPINGS = [qaoa.cycle.wires_to_edges(digraph_complete)] * 2

MWC = list(zip(DIGRAPHS, MWC_CONSTRAINED, COST_HAMILTONIANS, MIXER_HAMILTONIANS, MAPPINGS))


def decompose_hamiltonian(hamiltonian):

    coeffs = list(qml.math.toarray(hamiltonian.coeffs))
    ops = [i.name for i in hamiltonian.ops]
    wires = [i.wires for i in hamiltonian.ops]

    return [coeffs, ops, wires]


class TestCostHamiltonians:
    """Tests that the cost Hamiltonians are being generated correctly"""

    """Tests the cost Hamiltonian components"""

    def test_bit_driver_error(self):
        """Tests that the bit driver Hamiltonian throws the correct error"""

        with pytest.raises(ValueError, match=r"'b' must be either 0 or 1"):
            qaoa.bit_driver(range(3), 2)

    def test_bit_driver_output(self):
        """Tests that the bit driver Hamiltonian has the correct output"""

        H = qaoa.bit_driver(range(3), 1)
        hamiltonian = qml.Hamiltonian([1, 1, 1], [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)])

        assert decompose_hamiltonian(H) == decompose_hamiltonian(hamiltonian)

    def test_edge_driver_errors(self):
        """Tests that the edge driver Hamiltonian throws the correct errors"""

        with pytest.raises(
            ValueError, match=r"Encountered invalid entry in 'reward', expected 2-bit bitstrings."
        ):
            qaoa.edge_driver(g1, ["10", "11", 21, "g"])

        with pytest.raises(
            ValueError,
            match=r"'reward' cannot contain either '10' or '01', must contain neither or both.",
        ):
            qaoa.edge_driver(g1, ["11", "00", "01"])

        with pytest.raises(ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph"):
            qaoa.edge_driver([(0, 1), (1, 2)], ["00", "11"])

    @pytest.mark.parametrize(("graph", "reward", "hamiltonian"), EDGE_DRIVER)
    def test_edge_driver_output(self, graph, reward, hamiltonian):
        """Tests that the edge driver Hamiltonian throws the correct errors"""

        H = qaoa.edge_driver(graph, reward)
        assert decompose_hamiltonian(H) == decompose_hamiltonian(hamiltonian)

    """Tests the cost Hamiltonians"""

    def test_max_weight_cycle_errors(self):
        """Tests that the max weight cycle Hamiltonian throws the correct errors"""

        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph"
        ):
            qaoa.max_weight_cycle([(0, 1), (1, 2)])

    def test_cost_graph_error(self):
        """Tests that the cost Hamiltonians throw the correct error"""

        graph = [(0, 1), (1, 2)]

        with pytest.raises(ValueError, match=r"Input graph must be a nx\.Graph or rx\.PyGraph"):
            qaoa.maxcut(graph)
        with pytest.raises(ValueError, match=r"Input graph must be a nx\.Graph or rx\.PyGraph"):
            qaoa.max_independent_set(graph)
        with pytest.raises(ValueError, match=r"Input graph must be a nx\.Graph or rx\.PyGraph"):
            qaoa.min_vertex_cover(graph)
        with pytest.raises(ValueError, match=r"Input graph must be a nx\.Graph or rx\.PyGraph"):
            qaoa.max_clique(graph)

    @pytest.mark.parametrize(("graph", "cost_hamiltonian", "mixer_hamiltonian"), MAXCUT)
    def test_maxcut_output(self, graph, cost_hamiltonian, mixer_hamiltonian):
        """Tests that the output of the MaxCut method is correct"""

        cost_h, mixer_h = qaoa.maxcut(graph)

        assert decompose_hamiltonian(cost_hamiltonian) == decompose_hamiltonian(cost_h)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)

    @pytest.mark.parametrize("constrained", [True, False])
    def test_maxcut_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        graph = MAXCUT[0][0]
        cost_h, _ = qaoa.maxcut(graph)

        # check that all observables commute
        assert all(qml.is_commuting(o, cost_h.ops[0]) for o in cost_h.ops[1:])
        # check that the 1-group grouping information was set
        assert cost_h.grouping_indices is not None
        assert cost_h.grouping_indices == [list(range(len(cost_h.ops)))]

    @pytest.mark.parametrize(("graph", "constrained", "cost_hamiltonian", "mixer_hamiltonian"), MIS)
    def test_mis_output(self, graph, constrained, cost_hamiltonian, mixer_hamiltonian):
        """Tests that the output of the Max Indepenent Set method is correct"""

        cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=constrained)

        assert decompose_hamiltonian(cost_hamiltonian) == decompose_hamiltonian(cost_h)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)

    @pytest.mark.parametrize("constrained", [True, False])
    def test_mis_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        graph = MIS[0][0]
        cost_h, _ = qaoa.max_independent_set(graph)

        # check that all observables commute
        assert all(qml.is_commuting(o, cost_h.ops[0]) for o in cost_h.ops[1:])
        # check that the 1-group grouping information was set
        assert cost_h.grouping_indices is not None
        assert cost_h.grouping_indices == [list(range(len(cost_h.ops)))]

    @pytest.mark.parametrize(("graph", "constrained", "cost_hamiltonian", "mixer_hamiltonian"), MVC)
    def test_mvc_output(self, graph, constrained, cost_hamiltonian, mixer_hamiltonian):
        """Tests that the output of the Min Vertex Cover method is correct"""

        cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=constrained)

        assert decompose_hamiltonian(cost_hamiltonian) == decompose_hamiltonian(cost_h)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)

    @pytest.mark.parametrize("constrained", [True, False])
    def test_mvc_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        graph = MVC[0][0]
        cost_h, _ = qaoa.min_vertex_cover(graph)

        # check that all observables commute
        assert all(qml.is_commuting(o, cost_h.ops[0]) for o in cost_h.ops[1:])
        # check that the 1-group grouping information was set
        assert cost_h.grouping_indices is not None
        assert cost_h.grouping_indices == [list(range(len(cost_h.ops)))]

    @pytest.mark.parametrize(
        ("graph", "constrained", "cost_hamiltonian", "mixer_hamiltonian"), MAXCLIQUE
    )
    def test_max_clique_output(self, graph, constrained, cost_hamiltonian, mixer_hamiltonian):
        """Tests that the output of the Maximum Clique method is correct"""

        cost_h, mixer_h = qaoa.max_clique(graph, constrained=constrained)

        assert decompose_hamiltonian(cost_hamiltonian) == decompose_hamiltonian(cost_h)
        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)

    @pytest.mark.parametrize("constrained", [True, False])
    def test_max_clique_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        graph = MAXCLIQUE[0][0]
        cost_h, _ = qaoa.max_clique(graph)

        # check that all observables commute
        assert all(qml.is_commuting(o, cost_h.ops[0]) for o in cost_h.ops[1:])
        # check that the 1-group grouping information was set
        assert cost_h.grouping_indices is not None
        assert cost_h.grouping_indices == [list(range(len(cost_h.ops)))]

    @pytest.mark.parametrize(
        ("graph", "constrained", "cost_hamiltonian", "mixer_hamiltonian", "mapping"), MWC
    )
    def test_max_weight_cycle_output(
        self, graph, constrained, cost_hamiltonian, mixer_hamiltonian, mapping
    ):
        """Tests that the output of the maximum weighted cycle method is correct"""

        cost_h, mixer_h, m = qaoa.max_weight_cycle(graph, constrained=constrained)

        assert mapping == m

        c1, t1, w1 = decompose_hamiltonian(cost_hamiltonian)
        c2, t2, w2 = decompose_hamiltonian(cost_h)

        # There may be a very small numeric difference in the coeffs
        assert np.allclose(c1, c2)
        assert t1 == t2
        assert w1 == w2

        assert decompose_hamiltonian(mixer_hamiltonian) == decompose_hamiltonian(mixer_h)

    @pytest.mark.parametrize("constrained", [True, False])
    def test_max_weight_cycle_grouping(self, constrained):
        """Tests that the grouping information is set and correct"""

        graph = MWC[0][0]
        cost_h, _, _ = qaoa.max_weight_cycle(graph)

        # check that all observables commute
        assert all(qml.is_commuting(o, cost_h.ops[0]) for o in cost_h.ops[1:])
        # check that the 1-group grouping information was set
        assert cost_h.grouping_indices is not None
        assert cost_h.grouping_indices == [list(range(len(cost_h.ops)))]


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

        with pytest.raises(
            ValueError,
            match=r"hamiltonian must be written only in terms of PauliZ and Identity gates",
        ):
            qaoa.cost_layer(0.1, hamiltonian)

    @pytest.mark.parametrize(
        ("mixer", "gates"),
        [
            [
                qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliX(1)]),
                [qml.PauliRot(2, "X", wires=[0]), qml.PauliRot(2, "X", wires=[1])],
            ],
            [
                qaoa.xy_mixer(Graph([(0, 1), (1, 2), (2, 0)])),
                [
                    qml.PauliRot(1, "XX", wires=[0, 1]),
                    qml.PauliRot(1, "YY", wires=[0, 1]),
                    qml.PauliRot(1, "XX", wires=[0, 2]),
                    qml.PauliRot(1, "YY", wires=[0, 2]),
                    qml.PauliRot(1, "XX", wires=[1, 2]),
                    qml.PauliRot(1, "YY", wires=[1, 2]),
                ],
            ],
        ],
    )
    def test_mixer_layer_output(self, mixer, gates):
        """Tests that the gates of the mixer layer are correct"""

        alpha = 1

        with qml.tape.OperationRecorder() as rec:
            qaoa.mixer_layer(alpha, mixer)

        rec = rec.expand()

        for i, j in zip(rec.operations, gates):

            prep = [i.name, i.parameters, i.wires]
            target = [j.name, j.parameters, j.wires]

            assert prep == target

    @pytest.mark.parametrize(
        ("cost", "gates"),
        [
            [
                qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)]),
                [qml.PauliRot(2, "Z", wires=[0]), qml.PauliRot(2, "Z", wires=[1])],
            ],
            [
                qaoa.maxcut(Graph([(0, 1), (1, 2), (2, 0)]))[0],
                [
                    qml.PauliRot(1, "ZZ", wires=[0, 1]),
                    qml.PauliRot(1, "ZZ", wires=[0, 2]),
                    qml.PauliRot(1, "ZZ", wires=[1, 2]),
                ],
            ],
        ],
    )
    def test_cost_layer_output(self, cost, gates):
        """Tests that the gates of the cost layer is correct"""

        gamma = 1

        with qml.tape.OperationRecorder() as rec:
            qaoa.cost_layer(gamma, cost)

        rec = rec.expand()

        for i, j in zip(rec.operations, gates):
            prep = [i.name, i.parameters, i.wires]
            target = [j.name, j.parameters, j.wires]

        assert prep == target


class TestIntegration:
    """Test integration of the QAOA module with PennyLane"""

    def test_module_example(self, tol):
        """Test the example in the QAOA module docstring"""

        # Defines the wires and the graph on which MaxCut is being performed
        wires = range(3)
        graph = Graph([(0, 1), (1, 2), (2, 0)])

        # Defines the QAOA cost and mixer Hamiltonians
        cost_h, mixer_h = qaoa.maxcut(graph)

        # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(alpha, mixer_h)

        # Repeatedly applies layers of the QAOA ansatz
        def circuit(params, **kwargs):
            for w in wires:
                qml.Hadamard(wires=w)

            qml.layer(qaoa_layer, 2, params[0], params[1])

        # Defines the device and the QAOA cost function
        dev = qml.device("default.qubit", wires=len(wires))
        cost_function = catch_warn_ExpvalCost(circuit, cost_h, dev)

        res = cost_function([[1, 1], [1, 1]])
        expected = -1.8260274380964299

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_module_example_rx(self, tol):
        """Test the example in the QAOA module docstring"""

        # Defines the wires and the graph on which MaxCut is being performed
        wires = range(3)

        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1, ""), (1, 2, ""), (2, 0, "")])

        # Defines the QAOA cost and mixer Hamiltonians
        cost_h, mixer_h = qaoa.maxcut(graph)

        # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(alpha, mixer_h)

        # Repeatedly applies layers of the QAOA ansatz
        def circuit(params, **kwargs):
            for w in wires:
                qml.Hadamard(wires=w)

            qml.layer(qaoa_layer, 2, params[0], params[1])

        # Defines the device and the QAOA cost function
        dev = qml.device("default.qubit", wires=len(wires))
        cost_function = catch_warn_ExpvalCost(circuit, cost_h, dev)

        res = cost_function([[1, 1], [1, 1]])
        expected = -1.8260274380964299

        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestCycles:
    """Tests that ``cycle`` module functions are behaving correctly"""

    @pytest.mark.parametrize("g", [nx.lollipop_graph(4, 1), lollipop_graph_rx(4, 1)])
    def test_edges_to_wires(self, g):
        """Test that edges_to_wires returns the correct mapping"""
        r = edges_to_wires(g)

        assert r == {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5, (3, 4): 6}

    def test_edges_to_wires_error(self):
        """Test that edges_to_wires raises ValueError"""
        g = [1, 1, 1, 1]
        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph"
        ):
            edges_to_wires(g)

    def test_edges_to_wires_rx(self):
        """Test that edges_to_wires returns the correct mapping"""
        g = rx.generators.directed_mesh_graph(4, [0, 1, 2, 3])
        r = edges_to_wires(g)

        assert r == {
            (0, 1): 0,
            (0, 2): 1,
            (0, 3): 2,
            (1, 0): 3,
            (1, 2): 4,
            (1, 3): 5,
            (2, 0): 6,
            (2, 1): 7,
            (2, 3): 8,
            (3, 0): 9,
            (3, 1): 10,
            (3, 2): 11,
        }

    @pytest.mark.parametrize("g", [nx.lollipop_graph(4, 1), lollipop_graph_rx(4, 1)])
    def test_wires_to_edges(self, g):
        """Test that wires_to_edges returns the correct mapping"""
        r = wires_to_edges(g)

        assert r == {0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (1, 2), 4: (1, 3), 5: (2, 3), 6: (3, 4)}

    def test_wires_to_edges_error(self):
        """Test that wires_to_edges raises ValueError"""
        g = [1, 1, 1, 1]
        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph"
        ):
            wires_to_edges(g)

    def test_wires_to_edges_rx(self):
        """Test that wires_to_edges returns the correct mapping"""
        g = rx.generators.directed_mesh_graph(4, [0, 1, 2, 3])
        r = wires_to_edges(g)

        assert r == {
            0: (0, 1),
            1: (0, 2),
            2: (0, 3),
            3: (1, 0),
            4: (1, 2),
            5: (1, 3),
            6: (2, 0),
            7: (2, 1),
            8: (2, 3),
            9: (3, 0),
            10: (3, 1),
            11: (3, 2),
        }

    @pytest.mark.parametrize(
        "g",
        [nx.complete_graph(4).to_directed(), rx.generators.directed_mesh_graph(4, [0, 1, 2, 3])],
    )
    def test_partial_cycle_mixer_complete(self, g):
        """Test if the _partial_cycle_mixer function returns the expected Hamiltonian for a fixed
        example"""
        edge = (0, 1)

        h = _partial_cycle_mixer(g, edge)

        ops_expected = [
            qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(7),
            qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliX(7),
            qml.PauliY(0) @ qml.PauliX(1) @ qml.PauliY(7),
            qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(7),
            qml.PauliX(0) @ qml.PauliX(2) @ qml.PauliX(10),
            qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliX(10),
            qml.PauliY(0) @ qml.PauliX(2) @ qml.PauliY(10),
            qml.PauliX(0) @ qml.PauliY(2) @ qml.PauliY(10),
        ]
        coeffs_expected = [0.25, 0.25, 0.25, -0.25, 0.25, 0.25, 0.25, -0.25]

        assert h.coeffs == coeffs_expected
        assert all(op.wires == op_e.wires for op, op_e in zip(h.ops, ops_expected))
        assert all(op.name == op_e.name for op, op_e in zip(h.ops, ops_expected))

    @pytest.mark.parametrize(
        "g",
        [nx.complete_graph(4).to_directed(), rx.generators.directed_mesh_graph(4, [0, 1, 2, 3])],
    )
    def test_partial_cycle_mixer_incomplete(self, g):
        """Test if the _partial_cycle_mixer function returns the expected Hamiltonian for a fixed
        example"""
        g.remove_edge(2, 1)  # remove an egde to make graph incomplete
        edge = (0, 1)

        h = _partial_cycle_mixer(g, edge)

        ops_expected = [
            qml.PauliX(0) @ qml.PauliX(2) @ qml.PauliX(9),
            qml.PauliY(0) @ qml.PauliY(2) @ qml.PauliX(9),
            qml.PauliY(0) @ qml.PauliX(2) @ qml.PauliY(9),
            qml.PauliX(0) @ qml.PauliY(2) @ qml.PauliY(9),
        ]
        coeffs_expected = [0.25, 0.25, 0.25, -0.25]

        assert h.coeffs == coeffs_expected
        assert all(op.wires == op_e.wires for op, op_e in zip(h.ops, ops_expected))
        assert all(op.name == op_e.name for op, op_e in zip(h.ops, ops_expected))

    @pytest.mark.parametrize("g", [nx.complete_graph(4), rx.generators.mesh_graph(4, [0, 1, 2, 3])])
    def test_partial_cycle_mixer_error(self, g):
        """Test if the _partial_cycle_mixer raises ValueError"""
        g.remove_edge(2, 1)  # remove an egde to make graph incomplete
        edge = (0, 1)

        # Find Hamiltonian and its matrix representation
        with pytest.raises(ValueError, match="Input graph must be a nx.DiGraph or rx.PyDiGraph"):
            _partial_cycle_mixer(g, edge)

    @pytest.mark.parametrize(
        "g",
        [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])],
    )
    def test_cycle_mixer(self, g):
        """Test if the cycle_mixer Hamiltonian maps valid cycles to valid cycles"""

        n_nodes = 3
        m = wires_to_edges(g)
        n_wires = len(graph.edge_list() if isinstance(graph, rx.PyDiGraph) else graph.edges)

        # Find Hamiltonian and its matrix representation
        h = cycle_mixer(g)
        h_matrix = np.real_if_close(matrix(h, n_wires).toarray())

        # Decide which bitstrings are valid and which are invalid
        valid_bitstrings_indx = []
        invalid_bitstrings_indx = []

        for indx, bitstring in enumerate(itertools.product([0, 1], repeat=n_wires)):
            wires = [i for i, bit in enumerate(bitstring) if bit == 1]
            edges = [m[wire] for wire in wires]

            flows = [0 for i in range(n_nodes)]

            for start, end in edges:
                flows[start] += 1
                flows[end] -= 1

            # A bitstring is valid if the net flow is zero and we aren't the empty set or the set
            # of all edges. Note that the max out-flow constraint is not imposed, which means we can
            # pass through nodes more than once
            if sum(np.abs(flows)) == 0 and 0 < len(edges) < n_wires:
                valid_bitstrings_indx.append(indx)
            else:
                invalid_bitstrings_indx.append(indx)

        # Check that valid bitstrings map to a subset of the valid bitstrings
        for indx in valid_bitstrings_indx:
            column = h_matrix[:, indx]
            destination_indxs = set(np.argwhere(column != 0).flatten())

            assert destination_indxs.issubset(valid_bitstrings_indx)

        # Check that invalid bitstrings map to a subset of the invalid bitstrings
        for indx in invalid_bitstrings_indx:
            column = h_matrix[:, indx]
            destination_indxs = set(np.argwhere(column != 0).flatten())

            assert destination_indxs.issubset(invalid_bitstrings_indx)

        # Now consider a unitary generated by the Hamiltonian
        h_matrix_e = expm(1j * h_matrix)

        # We expect non-zero transitions among the set of valid bitstrings, and no transitions
        # outside
        for indx in valid_bitstrings_indx:
            column = h_matrix_e[:, indx]
            destination_indxs = np.argwhere(column != 0).flatten().tolist()
            assert destination_indxs == valid_bitstrings_indx

        # Check that invalid bitstrings transition within the set of invalid bitstrings
        for indx in invalid_bitstrings_indx:
            column = h_matrix_e[:, indx]
            destination_indxs = set(np.argwhere(column != 0).flatten().tolist())
            assert destination_indxs.issubset(invalid_bitstrings_indx)

    @pytest.mark.parametrize(
        "g",
        [nx.complete_graph(3), rx.generators.mesh_graph(3, [0, 1, 2])],
    )
    def test_cycle_mixer_error(self, g):
        """Test if the cycle_mixer raises ValueError"""
        # Find Hamiltonian and its matrix representation
        with pytest.raises(ValueError, match="Input graph must be a nx.DiGraph or rx.PyDiGraph"):
            cycle_mixer(g)

    @pytest.mark.parametrize("g", [nx.lollipop_graph(3, 1), lollipop_graph_rx(3, 1)])
    def test_matrix(self, g):
        """Test that the matrix function works as expected on a fixed example"""
        h = qml.qaoa.bit_flip_mixer(g, 0)

        mat = matrix(h, 4)
        mat_expected = np.array(
            [
                [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        assert np.allclose(mat.toarray(), mat_expected)

    def test_matrix_rx(self):
        """Test that the matrix function works as expected on a fixed example"""
        g = rx.generators.star_graph(4, [0, 1, 2, 3])
        h = qml.qaoa.bit_flip_mixer(g, 0)

        mat = matrix(h, 4)
        mat_expected = np.array(
            [
                [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        assert np.allclose(mat.toarray(), mat_expected)

    @pytest.mark.parametrize(
        "g", [nx.lollipop_graph(4, 1).to_directed(), lollipop_graph_rx(4, 1, to_directed=True)]
    )
    def test_edges_to_wires_directed(self, g):
        """Test that edges_to_wires returns the correct mapping on a directed graph"""
        r = edges_to_wires(g)

        assert r == {
            (0, 1): 0,
            (0, 2): 1,
            (0, 3): 2,
            (1, 0): 3,
            (1, 2): 4,
            (1, 3): 5,
            (2, 0): 6,
            (2, 1): 7,
            (2, 3): 8,
            (3, 0): 9,
            (3, 1): 10,
            (3, 2): 11,
            (3, 4): 12,
            (4, 3): 13,
        }

    @pytest.mark.parametrize(
        "g", [nx.lollipop_graph(4, 1).to_directed(), lollipop_graph_rx(4, 1, to_directed=True)]
    )
    def test_wires_to_edges_directed(self, g):
        """Test that wires_to_edges returns the correct mapping on a directed graph"""
        r = wires_to_edges(g)

        assert r == {
            0: (0, 1),
            1: (0, 2),
            2: (0, 3),
            3: (1, 0),
            4: (1, 2),
            5: (1, 3),
            6: (2, 0),
            7: (2, 1),
            8: (2, 3),
            9: (3, 0),
            10: (3, 1),
            11: (3, 2),
            12: (3, 4),
            13: (4, 3),
        }

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_loss_hamiltonian_complete(self, g):
        """Test if the loss_hamiltonian function returns the expected result on a
        manually-calculated example of a 3-node complete digraph"""
        if isinstance(g, rx.PyDiGraph):
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(sorted(g.edge_list()))}
            for k, v in edge_weight_data.items():
                g.update_edge(k[0], k[1], {"weight": v})
        else:
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(g.edges)}
            for k, v in edge_weight_data.items():
                g[k[0]][k[1]]["weight"] = v

        h = loss_hamiltonian(g)

        expected_ops = [
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(2),
            qml.PauliZ(3),
            qml.PauliZ(4),
            qml.PauliZ(5),
        ]
        expected_coeffs = [np.log(0.5), np.log(1), np.log(1.5), np.log(2), np.log(2.5), np.log(3)]

        assert np.allclose(expected_coeffs, h.coeffs)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])
        assert all([type(op) is type(exp) for op, exp in zip(h.ops, expected_ops)])

    def test_loss_hamiltonian_error(self):
        """Test if the loss_hamiltonian function raises ValueError"""
        with pytest.raises(
            ValueError, match=r"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph"
        ):
            loss_hamiltonian([(0, 1), (1, 2), (0, 2)])

    @pytest.mark.parametrize(
        "g", [nx.lollipop_graph(4, 1).to_directed(), lollipop_graph_rx(4, 1, to_directed=True)]
    )
    def test_loss_hamiltonian_incomplete(self, g):
        """Test if the loss_hamiltonian function returns the expected result on a
        manually-calculated example of a 4-node incomplete digraph"""
        if isinstance(g, rx.PyDiGraph):
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(sorted(g.edge_list()))}
            for k, v in edge_weight_data.items():
                g.update_edge(k[0], k[1], {"weight": v})
        else:
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(g.edges)}
            for k, v in edge_weight_data.items():
                g[k[0]][k[1]]["weight"] = v
        h = loss_hamiltonian(g)

        expected_ops = [
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(2),
            qml.PauliZ(3),
            qml.PauliZ(4),
            qml.PauliZ(5),
            qml.PauliZ(6),
            qml.PauliZ(7),
            qml.PauliZ(8),
            qml.PauliZ(9),
            qml.PauliZ(10),
            qml.PauliZ(11),
            qml.PauliZ(12),
            qml.PauliZ(13),
        ]
        expected_coeffs = [
            np.log(0.5),
            np.log(1),
            np.log(1.5),
            np.log(2),
            np.log(2.5),
            np.log(3),
            np.log(3.5),
            np.log(4),
            np.log(4.5),
            np.log(5),
            np.log(5.5),
            np.log(6),
            np.log(6.5),
            np.log(7),
        ]

        assert np.allclose(expected_coeffs, h.coeffs)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])
        assert all([type(op) is type(exp) for op, exp in zip(h.ops, expected_ops)])

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_self_loop_raises_error(self, g):
        """Test graphs with self loop raises ValueError"""

        if isinstance(g, rx.PyDiGraph):
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(g.edges())}
            for k, v in complete_edge_weight_data.items():
                g.update_edge(k[0], k[1], {"weight": v})
            g.add_edge(1, 1, "")  # add self loop

        else:
            edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(g.edges)}
            for k, v in edge_weight_data.items():
                g[k[0]][k[1]]["weight"] = v
            g.add_edge(1, 1)  # add self loop

        with pytest.raises(ValueError, match="Graph contains self-loops"):
            loss_hamiltonian(g)

    def test_missing_edge_weight_data_raises_error(self):
        """Test graphs with no edge weight data raises `KeyError`"""
        g = nx.complete_graph(3).to_directed()
        with pytest.raises(KeyError, match="does not contain weight data"):
            loss_hamiltonian(g)

    def test_missing_edge_weight_data_without_weights(self):
        """Test graphs with no edge weight data raises `KeyError`"""
        g = rx.generators.mesh_graph(3, [0, 1, 2])
        with pytest.raises(TypeError, match="does not contain weight data"):
            loss_hamiltonian(g)

    def test_square_hamiltonian_terms(self):
        """Test if the _square_hamiltonian_terms function returns the expected result on a fixed
        example"""
        coeffs = [1, -1, -1, 1]
        ops = [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(3)]

        expected_coeffs = [
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
        ]
        expected_ops = [
            qml.Identity(0),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(3),
            qml.PauliZ(0),
            qml.Identity(0),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(3),
            qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.Identity(0),
            qml.PauliZ(1) @ qml.PauliZ(3),
            qml.PauliZ(3),
            qml.PauliZ(0) @ qml.PauliZ(3),
            qml.PauliZ(1) @ qml.PauliZ(3),
            qml.Identity(0),
        ]

        squared_coeffs, squared_ops = _square_hamiltonian_terms(coeffs, ops)

        assert squared_coeffs == expected_coeffs
        assert all(
            [
                op1.name == op2.name and op1.wires == op2.wires
                for op1, op2 in zip(expected_ops, squared_ops)
            ]
        )

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_inner_out_flow_constraint_hamiltonian(self, g):
        """Test if the _inner_out_flow_constraint_hamiltonian function returns the expected result
        on a manually-calculated example of a 3-node complete digraph relative to the 0 node"""
        h = _inner_out_flow_constraint_hamiltonian(g, 0)

        expected_ops = [
            qml.Identity(0),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0),
            qml.PauliZ(1),
        ]

        expected_coeffs = [2, 2, -2, -2]

        assert np.allclose(expected_coeffs, h.coeffs)
        for i, expected_op in enumerate(expected_ops):
            assert str(h.ops[i]) == str(expected_op)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])

    @pytest.mark.parametrize("g", [nx.complete_graph(3), rx.generators.mesh_graph(3, [0, 1, 2])])
    def test_inner_out_flow_constraint_hamiltonian_error(self, g):
        """Test if the _inner_out_flow_constraint_hamiltonian function raises ValueError"""
        with pytest.raises(ValueError, match=r"Input graph must be a nx.DiGraph or rx.PyDiGraph"):
            _inner_out_flow_constraint_hamiltonian(g, 0)

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_inner_net_flow_constraint_hamiltonian(self, g):
        """Test if the _inner_net_flow_constraint_hamiltonian function returns the expected result on a manually-calculated
        example of a 3-node complete digraph relative to the 0 node"""
        h = _inner_net_flow_constraint_hamiltonian(g, 0)

        expected_ops = [
            qml.Identity(0),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(2),
            qml.PauliZ(0) @ qml.PauliZ(4),
            qml.PauliZ(1) @ qml.PauliZ(2),
            qml.PauliZ(1) @ qml.PauliZ(4),
            qml.PauliZ(2) @ qml.PauliZ(4),
        ]
        expected_coeffs = [4, 2, -2, -2, -2, -2, 2]

        assert np.allclose(expected_coeffs, h.coeffs)
        for i, expected_op in enumerate(expected_ops):
            assert str(h.ops[i]) == str(expected_op)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])

    @pytest.mark.parametrize("g", [nx.complete_graph(3), rx.generators.mesh_graph(3, [0, 1, 2])])
    def test_inner_net_flow_constraint_hamiltonian_error(self, g):
        """Test if the _inner_net_flow_constraint_hamiltonian function returns raises ValueError"""
        with pytest.raises(ValueError, match=r"Input graph must be a nx.DiGraph or rx.PyDiGraph"):
            _inner_net_flow_constraint_hamiltonian(g, 0)

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_inner_out_flow_constraint_hamiltonian_non_complete(self, g):
        """Test if the _inner_out_flow_constraint_hamiltonian function returns the expected result
        on a manually-calculated example of a 3-node complete digraph relative to the 0 node, with
        the (0, 1) edge removed"""
        g.remove_edge(0, 1)
        h = _inner_out_flow_constraint_hamiltonian(g, 0)

        expected_ops = [qml.PauliZ(wires=[0])]
        expected_coeffs = [0]

        assert np.allclose(expected_coeffs, h.coeffs)
        for i, expected_op in enumerate(expected_ops):
            assert str(h.ops[i]) == str(expected_op)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_inner_net_flow_constraint_hamiltonian_non_complete(self, g):
        """Test if the _inner_net_flow_constraint_hamiltonian function returns the expected result on a manually-calculated
        example of a 3-node complete digraph relative to the 0 node, with the (1, 0) edge removed"""
        g.remove_edge(1, 0)
        h = _inner_net_flow_constraint_hamiltonian(g, 0)

        expected_ops = [
            qml.Identity(0),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(3),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(0) @ qml.PauliZ(3),
            qml.PauliZ(1) @ qml.PauliZ(3),
        ]
        expected_coeffs = [4, -2, -2, 2, 2, -2, -2]

        assert np.allclose(expected_coeffs, h.coeffs)
        for i, expected_op in enumerate(expected_ops):
            assert str(h.ops[i]) == str(expected_op)
        assert all([op.wires == exp.wires for op, exp in zip(h.ops, expected_ops)])

    def test_out_flow_constraint_raises(self, monkeypatch):
        """Test the out-flow constraint function may raise an error."""

        class OtherDirectedGraph(nx.DiGraph):
            def __init__(self, *args, **kwargs):
                pass

        g = OtherDirectedGraph()
        with pytest.raises(ValueError, match="must be directed"):
            out_flow_constraint(g)

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_out_flow_constraint(self, g):
        """Test the out-flow constraint Hamiltonian is minimised by states that correspond to
        subgraphs that only ever have 0 or 1 edge leaving each node
        """
        h = out_flow_constraint(g)
        m = wires_to_edges(g)
        wires = len(g.edge_list() if isinstance(g, rx.PyDiGraph) else g.edges)

        # We use PL to find the energies corresponding to each possible bitstring
        dev = qml.device("default.qubit", wires=wires)

        def states(basis_state, **kwargs):
            qml.BasisState(basis_state, wires=range(wires))

        cost = catch_warn_ExpvalCost(states, h, dev, optimize=True)

        # Calculate the set of all bitstrings
        bitstrings = itertools.product([0, 1], repeat=wires)

        # Calculate the corresponding energies
        energies_bitstrings = (
            (cost(np.array(bitstring)).numpy(), bitstring) for bitstring in bitstrings
        )

        for energy, bs in energies_bitstrings:

            # convert binary string to wires then wires to edges
            wires_ = tuple(i for i, s in enumerate(bs) if s != 0)
            edges = tuple(m[w] for w in wires_)

            # find the number of edges leaving each node
            if isinstance(g, rx.PyDiGraph):
                num_edges_leaving_node = {node: 0 for node in g.nodes()}
            else:
                num_edges_leaving_node = {node: 0 for node in g.nodes}
            for e in edges:
                num_edges_leaving_node[e[0]] += 1

            # check that if the max number of edges is <=1 it corresponds to a state that minimizes
            # the out_flow_constraint Hamiltonian
            if max(num_edges_leaving_node.values()) > 1:
                assert energy > min(energies_bitstrings)[0]
            elif max(num_edges_leaving_node.values()) <= 1:
                assert energy == min(energies_bitstrings)[0]

    @pytest.mark.parametrize("g", [nx.complete_graph(3), rx.generators.mesh_graph(3, [0, 1, 2])])
    def test_out_flow_constraint_undirected_raises_error(self, g):
        """Test `out_flow_constraint` raises ValueError if input graph is not directed"""
        with pytest.raises(ValueError):
            out_flow_constraint(g)

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_net_flow_constraint(self, g):
        """Test if the net_flow_constraint Hamiltonian is minimized by states that correspond to a
        collection of edges with zero flow"""
        h = net_flow_constraint(g)
        m = wires_to_edges(g)
        wires = len(g.edge_list() if isinstance(g, rx.PyDiGraph) else g.edges)

        # We use PL to find the energies corresponding to each possible bitstring
        dev = qml.device("default.qubit", wires=wires)

        def energy(basis_state, **kwargs):
            qml.BasisState(basis_state, wires=range(wires))

        cost = catch_warn_ExpvalCost(energy, h, dev, optimize=True)

        # Calculate the set of all bitstrings
        states = itertools.product([0, 1], repeat=wires)

        # Calculate the corresponding energies
        energies_states = ((cost(np.array(state)).numpy(), state) for state in states)

        # We now have the energies of each bitstring/state. We also want to calculate the net flow of
        # the corresponding edges
        for energy, state in energies_states:

            # This part converts from a binary string of wires selected to graph edges
            wires_ = tuple(i for i, s in enumerate(state) if s != 0)
            edges = tuple(m[w] for w in wires_)

            # Calculates the number of edges entering and leaving a given node
            if isinstance(g, rx.PyDiGraph):
                in_flows = np.zeros(len(g.nodes()))
                out_flows = np.zeros(len(g.nodes()))
            else:
                in_flows = np.zeros(len(g.nodes))
                out_flows = np.zeros(len(g.nodes))

            for e in edges:
                in_flows[e[0]] += 1
                out_flows[e[1]] += 1

            net_flow = np.sum(np.abs(in_flows - out_flows))

            # The test requires that a set of edges with zero net flow must have a corresponding
            # bitstring that minimized the energy of the Hamiltonian
            if net_flow == 0:
                assert energy == min(energies_states)[0]
            else:
                assert energy > min(energies_states)[0]

    @pytest.mark.parametrize("g", [nx.complete_graph(3), rx.generators.mesh_graph(3, [0, 1, 2])])
    def test_net_flow_constraint_wrong_graph_type_raises_error(self, g):
        """Test `net_flow_constraint` raises ValueError if input graph is not
        the correct graph type"""

        with pytest.raises(ValueError, match="Input graph must be"):
            h = net_flow_constraint(g)

    def test_net_flow_constraint_undirected_raises_error(self, monkeypatch):
        """Test the net-flow constraint function may raise an error."""

        class OtherDirectedGraph(nx.DiGraph):
            def __init__(self, *args, **kwargs):
                pass

        g = OtherDirectedGraph()
        with pytest.raises(ValueError, match="must be directed"):
            net_flow_constraint(g)

    @pytest.mark.parametrize(
        "g", [nx.complete_graph(3).to_directed(), rx.generators.directed_mesh_graph(3, [0, 1, 2])]
    )
    def test_net_flow_and_out_flow_constraint(self, g):
        """Test the combined net-flow and out-flow constraint Hamiltonian is minimised by states that correspond to subgraphs
        that qualify as simple_cycles
        """
        g = nx.complete_graph(3).to_directed()
        h = net_flow_constraint(g) + out_flow_constraint(g)
        m = wires_to_edges(g)
        wires = len(g.edge_list() if isinstance(g, rx.PyDiGraph) else g.edges)

        # Find the energies corresponding to each possible bitstring
        dev = qml.device("default.qubit", wires=wires)

        def states(basis_state, **kwargs):
            qml.BasisState(basis_state, wires=range(wires))

        cost = catch_warn_ExpvalCost(states, h, dev, optimize=True)

        # Calculate the set of all bitstrings
        bitstrings = itertools.product([0, 1], repeat=wires)

        # Calculate the corresponding energies
        energies_bitstrings = (
            (cost(np.array(bitstring)).numpy(), bitstring) for bitstring in bitstrings
        )

        def find_simple_cycle(list_of_edges):
            """Returns True if list_of_edges contains a permutation corresponding to a simple cycle"""
            permutations = list(itertools.permutations(list_of_edges))

            for edges in permutations:
                if edges[0][0] != edges[-1][-1]:  # check first node is equal to last node
                    continue
                all_nodes = []
                for edge in edges:
                    for n in edge:
                        all_nodes.append(n)
                inner_nodes = all_nodes[
                    1:-1
                ]  # find all nodes in all edges excluding the first and last nodes
                nodes_out = [
                    inner_nodes[i] for i in range(len(inner_nodes)) if i % 2 == 0
                ]  # find the nodes each edge is leaving
                node_in = [
                    inner_nodes[i] for i in range(len(inner_nodes)) if i % 2 != 0
                ]  # find the nodes each edge is entering
                if nodes_out == node_in and (
                    len([all_nodes[0]] + nodes_out) == len(set([all_nodes[0]] + nodes_out))
                ):  # check that each edge connect to the next via a common node and that no node is crossed more than once
                    return True

        for energy, bs in energies_bitstrings:
            # convert binary string to wires then wires to edges
            wires_ = tuple(i for i, s in enumerate(bs) if s != 0)
            edges = tuple(m[w] for w in wires_)

            if len(edges) and find_simple_cycle(edges):
                assert energy == min(energies_bitstrings)[0]
            elif len(edges) and not find_simple_cycle(edges):
                assert energy > min(energies_bitstrings)[0]
